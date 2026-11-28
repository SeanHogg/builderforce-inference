// Materialize the native subset of the Train Once contract from its TypeScript source.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [contractPath, compilerPath, mode] = process.argv.slice(2);
if (!contractPath || !compilerPath || (mode && mode !== '--check')) {
  throw new Error('Usage: node generate-experience-model.mjs <engine types.ts> <typescript.js> [--check]');
}
const { default: ts } = await import(pathToFileURL(resolve(compilerPath)).href);
const source = ts.createSourceFile(contractPath, readFileSync(contractPath, 'utf8'), ts.ScriptTarget.Latest, true);
const fieldsKeptForReplay = new Set(['id', 'name', 'program', 'args', 'params', 'steps']);
const structs = ['ElementRef', 'RecordedStep', 'Episode', 'SkillParam', 'SkillStep', 'Skill'];
const taggedUnions = new Map([['Action', 'kind'], ['SkillValue', 'from'], ['SkillAction', 'kind']]);
const enums = ['RunStatus', 'StepOutcome'];
const modelTypes = new Set([...structs, ...taggedUnions.keys(), ...enums]);
const order = ['ElementRef', 'Action', 'RecordedStep', 'Episode', 'SkillParam', 'SkillValue', 'SkillAction', 'SkillStep', 'Skill', ...enums];
const propertiesOf = (members) => members.filter(ts.isPropertySignature);

function rustName(name) {
  return name.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase());
}

function variantName(tag) {
  if (!/^[a-z][A-Za-z0-9]*$/.test(tag)) throw new Error('Unsupported variant: ' + tag);
  return tag[0].toUpperCase() + tag.slice(1);
}

function unionMembers(name, node) {
  if (!node || !ts.isTypeAliasDeclaration(node) || !ts.isUnionTypeNode(node.type)) {
    throw new Error('Expected a union for ' + name);
  }
  return node.type.types;
}

function stringLiteral(type) {
  if (!type || !ts.isLiteralTypeNode(type) || !ts.isStringLiteral(type.literal)) {
    throw new Error('Expected a string discriminator');
  }
  return type.literal.text;
}

function rustType(type, field) {
  if (ts.isArrayTypeNode(type)) return 'Vec<' + rustType(type.elementType, field) + '>';
  if (ts.isTypeReferenceNode(type) && modelTypes.has(type.typeName.getText()) && !type.typeArguments) {
    return type.typeName.getText();
  }
  if (type.kind === ts.SyntaxKind.StringKeyword) return 'String';
  if (type.kind === ts.SyntaxKind.BooleanKeyword) return 'bool';
  if (type.kind === ts.SyntaxKind.NumberKeyword) {
    if (field === 'relX' || field === 'relY') return 'i32';
    if (field === 'atMs') return 'u64';
    if (field === 'startedAt' || field === 'endedAt') return 'i64';
  }
  throw new Error('Unsupported native field ' + field + ': ' + type.getText());
}

function fieldDeclaration(member, owner, isStruct) {
  if (!member.type || !ts.isIdentifier(member.name)) throw new Error('Expected a typed property in ' + owner);
  const name = member.name.text;
  const parts = ts.isUnionTypeNode(member.type) ? member.type.types : [member.type];
  const substantive = parts.filter((type) => !(ts.isLiteralTypeNode(type) && type.literal.kind === ts.SyntaxKind.NullKeyword));
  if (substantive.length !== 1) throw new Error('Unsupported field union in ' + owner + '.' + name);
  const base = rustType(substantive[0], name);
  const optional = !!member.questionToken || parts.length !== substantive.length;
  const defaultsFalse = owner === 'Action' && name === 'secret' && base === 'bool';
  const type = optional && !defaultsFalse ? 'Option<' + base + '>' : base;
  const visibility = isStruct ? 'pub ' : '';
  const attributes = [];
  if (defaultsFalse || (isStruct && name === 'args')) attributes.push('#[serde(default)]');
  if (!isStruct && rustName(name) !== name) attributes.push('#[serde(rename = "' + name + '")]');
  return [...attributes, visibility + rustName(name) + ': ' + type + ','].join('\n');
}

function struct(name, node) {
  if (!node || !ts.isInterfaceDeclaration(node) || node.heritageClauses) throw new Error('Expected an interface for ' + name);
  let properties = propertiesOf(node.members);
  if (properties.length !== node.members.length) throw new Error('Unsupported member in ' + name);
  if (name === 'Skill') {
    properties = properties.filter((field) => fieldsKeptForReplay.has(field.name.getText()));
    if (properties.length !== fieldsKeptForReplay.size) throw new Error('Native Skill projection is missing a field');
  }
  const derives = 'Debug, Clone, PartialEq, Serialize, Deserialize' + (name === 'ElementRef' ? ', Default' : '');
  const serde = 'rename_all = "camelCase"' + (name === 'ElementRef' ? ', default' : '');
  const members = properties.map((field) => fieldDeclaration(field, name, true)).join('\n');
  return '#[derive(' + derives + ')]\n#[serde(' + serde + ')]\npub struct ' + name + ' {\n' + indent(members) + '\n}';
}

function taggedEnum(name, node, tag) {
  const variants = unionMembers(name, node).map((type) => {
    if (!ts.isTypeLiteralNode(type)) throw new Error('Expected a tagged object in ' + name);
    const properties = propertiesOf(type.members);
    if (properties.length !== type.members.length) throw new Error('Unsupported variant member in ' + name);
    const discriminant = properties.find((field) => field.name.getText() === tag);
    const fields = properties.filter((field) => field !== discriminant).map((field) => fieldDeclaration(field, name, false));
    const variant = variantName(stringLiteral(discriminant?.type));
    return variant + ' {\n' + indent(fields.join('\n')) + '\n},';
  });
  return '#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]\n#[serde(tag = "' + tag +
    '", rename_all = "camelCase")]\npub enum ' + name + ' {\n' + indent(variants.join('\n')) + '\n}';
}

function plainEnum(name, node) {
  const variants = unionMembers(name, node).map((type) => '    ' + variantName(stringLiteral(type)) + ',');
  return '#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]\n' +
    '#[serde(rename_all = "lowercase")]\npub enum ' + name + ' {\n' + variants.join('\n') + '\n}';
}

function indent(text) {
  return text.split('\n').map((line) => '    ' + line).join('\n');
}

const definitions = new Map(source.statements.filter((node) => node.name && modelTypes.has(node.name.getText()))
  .map((node) => [node.name.getText(), node]));
const generated = '// Generated by generate-experience-model.mjs from memory-engine/src/experience/types.ts.\n' +
  '// The native helper keeps only the fields it reads or writes.\n\n' +
  order.map((name) => {
    const node = definitions.get(name);
    if (structs.includes(name)) return struct(name, node);
    if (taggedUnions.has(name)) return taggedEnum(name, node, taggedUnions.get(name));
    return plainEnum(name, node);
  }).join('\n\n') + '\n';
const output = new URL('../src/experience_model.generated.rs', import.meta.url);
if (mode === '--check') {
  if (readFileSync(output, 'utf8').replace(/\r\n/g, '\n') !== generated) throw new Error('Native experience model needs regeneration');
} else {
  writeFileSync(output, generated);
}


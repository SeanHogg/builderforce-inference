//! Native methods for the generated Train Once wire contract. The memory engine
//! owns the schema. The helper reads a projection of each procedure for replay.

use serde::{Deserialize, Serialize};

include!("experience_model.generated.rs");

impl ElementRef {
    /// The label a person would use for it: its name, else its automation id (the
    /// engine's `elementLabel`).
    pub fn label(&self) -> &str {
        if !self.name.trim().is_empty() { self.name.trim() } else { self.automation_id.trim() }
    }
}

impl SkillAction {
    pub fn target(&self) -> Option<&ElementRef> {
        match self {
            SkillAction::Click { target } | SkillAction::SetValue { target, .. } => Some(target),
            SkillAction::Keys { target, .. } => target.as_ref(),
        }
    }
}

/// A short random id (16 hex chars), in the engine's `newExperienceId` alphabet — safe as
/// a folder name, which is what an episode id also is.
pub fn new_id() -> String {
    format!("{:016x}", rand::random::<u64>())
}

/// Epoch milliseconds — the engine's timestamp unit.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_engines_skill_json_and_ignores_what_it_does_not_use() {
        let json = r#"{"id":"s","name":"Invoice","program":"ledger.exe","args":[],"sourceEpisode":"e","createdAt":1,
            "routine":{"schedule":{"every":"minutes","minutes":5},"values":{},"enabled":true},
            "params":[{"name":"amount","label":"Amount","default":"250","secret":false}],
            "steps":[{"id":"a","requiresApproval":true,"screenshot":null,
                      "action":{"kind":"setValue","target":{"name":"Amount","controlType":"Edit"},"value":{"from":"param","name":"amount"}}}]}"#;
        let skill: Skill = serde_json::from_str(json).unwrap();
        assert_eq!(skill.params[0].name, "amount");
        assert!(matches!(&skill.steps[0].action, SkillAction::SetValue { value: SkillValue::Param { name }, .. } if name == "amount"));
        assert_eq!(skill.steps[0].action.target().unwrap().label(), "Amount");
    }

    #[test]
    fn writes_episodes_in_the_engines_shape() {
        let ep = Episode {
            id: new_id(),
            name: "n".into(),
            program: "p".into(),
            args: vec![],
            started_at: 1,
            ended_at: None,
            steps: vec![RecordedStep {
                id: "s1".into(),
                at_ms: 3,
                action: Action::SetValue { target: ElementRef::default(), value: String::new(), secret: true },
                screenshot: None,
            }],
        };
        let v = serde_json::to_value(&ep).unwrap();
        assert_eq!(v["startedAt"], 1);
        assert_eq!(v["steps"][0]["atMs"], 3);
        assert_eq!(v["steps"][0]["action"]["kind"], "setValue");
        assert_eq!(v["steps"][0]["action"]["secret"], true);
        assert_eq!(ep.id.len(), 16);
    }
}

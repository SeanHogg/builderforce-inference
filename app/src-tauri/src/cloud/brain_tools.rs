//! Platform tools available to a Synapse Brain turn.

use bf_cloud::tools::PlatformTool;
use bf_cloud::{CloudError, Session};
use serde_json::Value;

const SURFACE: &str = "delivery";

/// `builtin_tasks_create` → `tasks create`.
pub fn label(name: &str) -> String {
    name.strip_prefix("builtin_").unwrap_or(name).replace('_', " ")
}

pub struct Toolbox {
    tools: Vec<PlatformTool>,
}

impl Toolbox {
    pub fn gather(s: &Session) -> Result<Self, CloudError> {
        let tools = match s.platform_tools(SURFACE) {
            Ok(tools) => tools,
            Err(CloudError::KeyRejected) => return Err(CloudError::KeyRejected),
            Err(_) => Vec::new(),
        };
        Ok(Self { tools })
    }

    pub fn find(&self, name: &str) -> Option<&PlatformTool> {
        if let Some(tool) = self.tools.iter().find(|tool| tool.name == name) {
            return Some(tool);
        }
        let mut aliases = self.tools.iter().filter(|tool| tool.legacy_name.as_deref() == Some(name));
        let tool = aliases.next()?;
        aliases.next().is_none().then_some(tool)
    }

    pub fn offered(&self) -> Vec<Value> {
        self.tools.iter().map(PlatformTool::as_function).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_reads_as_what_it_does() {
        assert_eq!(label("builtin_tasks_create"), "tasks create");
        assert_eq!(label("acme_send_invoice"), "acme send invoice");
    }
}

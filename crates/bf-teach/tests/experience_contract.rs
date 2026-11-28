use bf_teach::model::{Episode, RunStatus, Skill, StepOutcome};
use serde_json::{Value, json};

#[test]
fn rust_round_trips_the_engine_contract() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/experience-contract.json"))
        .expect("Train Once engine output");
    let episode: Episode = serde_json::from_value(fixture["episode"].clone()).unwrap();
    assert_eq!(serde_json::to_value(episode).unwrap(), fixture["episode"]);

    let mut replay_payload = fixture["skill"].clone();
    let skill: Skill = serde_json::from_value(replay_payload.clone()).unwrap();
    let object = replay_payload.as_object_mut().unwrap();
    object.remove("sourceEpisode");
    object.remove("createdAt");
    object.remove("routine");
    assert_eq!(serde_json::to_value(skill).unwrap(), replay_payload);
}

#[test]
fn results_use_the_experience_ledger_tags() {
    let statuses = [RunStatus::Running, RunStatus::Succeeded, RunStatus::Failed, RunStatus::Stopped, RunStatus::Denied];
    let outcomes = [StepOutcome::Ok, StepOutcome::Fallback, StepOutcome::Failed, StepOutcome::Approved, StepOutcome::Denied];
    assert_eq!(json!(statuses), json!(["running", "succeeded", "failed", "stopped", "denied"]));
    assert_eq!(json!(outcomes), json!(["ok", "fallback", "failed", "approved", "denied"]));
}

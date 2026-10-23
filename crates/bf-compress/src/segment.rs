//! Splitting text into what compression may touch (prose, as words) and what it must
//! carry through byte-for-byte (fenced code blocks).

/// A run of the input: prose is compressed word by word, a fence is copied verbatim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Segment<'a> {
    Prose(&'a str),
    /// A ``` / ~~~ fenced block, from its opening fence line through its closing fence
    /// line (or to the end of the text when it is never closed).
    Fence(&'a str),
}

/// One prose word and the whitespace that preceded it.
#[derive(Debug, Clone, PartialEq)]
pub struct Word<'a> {
    pub gap: &'a str,
    pub text: &'a str,
}

/// The opening fence of a line: its marker char and run length, when the line (after at
/// most three spaces of indent, CommonMark's rule) starts with three or more of ` or ~.
fn fence_marker(line: &str) -> Option<(char, usize)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let ch = rest.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let run = rest.chars().take_while(|c| *c == ch).count();
    (run >= 3).then_some((ch, run))
}

/// Closes an open fence: the same marker char, at least as long, nothing else on the line.
fn closes(line: &str, open: (char, usize)) -> bool {
    match fence_marker(line) {
        Some((ch, run)) => ch == open.0 && run >= open.1 && line.trim().chars().all(|c| c == ch),
        None => false,
    }
}

/// The text as alternating prose and fenced-code segments, covering every byte in order.
pub fn segments(text: &str) -> Vec<Segment<'_>> {
    let mut out = Vec::new();
    let mut prose_start = 0usize;
    let mut fence: Option<(usize, (char, usize))> = None;
    let mut pos = 0usize;
    for line in text.split_inclusive('\n') {
        let bare = line.trim_end_matches(['\n', '\r']);
        match fence {
            None => {
                if let Some(marker) = fence_marker(bare) {
                    if pos > prose_start {
                        out.push(Segment::Prose(&text[prose_start..pos]));
                    }
                    fence = Some((pos, marker));
                }
            }
            Some((start, marker)) => {
                if closes(bare, marker) {
                    let end = pos + line.len();
                    out.push(Segment::Fence(&text[start..end]));
                    fence = None;
                    prose_start = end;
                }
            }
        }
        pos += line.len();
    }
    match fence {
        Some((start, _)) => out.push(Segment::Fence(&text[start..])),
        None if prose_start < text.len() => out.push(Segment::Prose(&text[prose_start..])),
        None => {}
    }
    out
}

/// Prose as words, each with the whitespace before it. Trailing whitespace is dropped.
pub fn words(prose: &str) -> Vec<Word<'_>> {
    let mut out = Vec::new();
    let mut gap_start = 0usize;
    let mut word_start: Option<usize> = None;
    for (i, c) in prose.char_indices() {
        match (c.is_whitespace(), word_start) {
            (false, None) => word_start = Some(i),
            (true, Some(ws)) => {
                out.push(Word { gap: &prose[gap_start..ws], text: &prose[ws..i] });
                word_start = None;
                gap_start = i;
            }
            _ => {}
        }
    }
    if let Some(ws) = word_start {
        out.push(Word { gap: &prose[gap_start..ws], text: &prose[ws..] });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_are_isolated_verbatim() {
        let t = "intro text\n```rust\nfn a() {}\n```\nafter\n";
        assert_eq!(
            segments(t),
            vec![Segment::Prose("intro text\n"), Segment::Fence("```rust\nfn a() {}\n```\n"), Segment::Prose("after\n")]
        );
    }

    #[test]
    fn unclosed_and_tilde_fences() {
        assert_eq!(segments("a\n~~~~\nx\n~~~\ny"), vec![Segment::Prose("a\n"), Segment::Fence("~~~~\nx\n~~~\ny")]);
        // A longer marker closes; a shorter or different one does not.
        let t = "````\n```\n````\nz";
        assert_eq!(segments(t), vec![Segment::Fence("````\n```\n````\n"), Segment::Prose("z")]);
        // Indented four spaces is not a fence.
        assert_eq!(segments("    ```\nq"), vec![Segment::Prose("    ```\nq")]);
    }

    #[test]
    fn segments_cover_every_byte() {
        let t = "p1\n```\nc\n```\np2\n```js\nopen";
        let joined: String = segments(t)
            .iter()
            .map(|s| match s {
                Segment::Prose(p) | Segment::Fence(p) => *p,
            })
            .collect();
        assert_eq!(joined, t);
    }

    #[test]
    fn words_keep_their_leading_gap() {
        let w = words("  the quick\n\nfox ");
        assert_eq!(
            w,
            vec![
                Word { gap: "  ", text: "the" },
                Word { gap: " ", text: "quick" },
                Word { gap: "\n\n", text: "fox" },
            ]
        );
        assert!(words(" \n ").is_empty());
    }
}

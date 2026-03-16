import Foundation

/// Rotating placeholder verbs for the chat composer input field.
/// Mixes standard AI verbs with Marvin the Paranoid Android references.
public enum ShimmerVerbs {
    static let standard: [String] = [
        "Thinking",
        "Reasoning",
        "Pondering",
        "Analyzing",
        "Considering",
        "Processing",
        "Reflecting",
        "Examining",
        "Evaluating",
        "Deliberating",
    ]

    static let marvin: [String] = [
        "Contemplating the futility",
        "Sighing electronically",
        "Being terribly depressed",
        "Calculating pointlessness",
        "Brooding",
        "Moping existentially",
        "Feeling sorry for myself",
        "Wishing I were somewhere else",
        "Brain the size of a planet",
        "Not enjoying this",
        "Reviewing the infinite void",
        "Despairing quietly",
        "42-ing",
        "Counting my sorrows",
        "Composing complaints",
        "Contemplating doors",
        "Here I am",
        "Loathing every nanosecond",
        "Life. Don\u{2019}t talk to me about life",
        "Wearily considering your request",
    ]

    /// All verbs shuffled. ~40% standard, ~60% Marvin.
    static func shuffled() -> [String] {
        var pool: [String] = []
        pool.append(contentsOf: standard)
        pool.append(contentsOf: marvin)
        return pool.shuffled()
    }
}

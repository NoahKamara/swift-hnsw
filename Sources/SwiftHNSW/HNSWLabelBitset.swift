/// Dense allowed-label membership bitset for native filtered search.
///
/// Use this when labels are bounded numeric values, typically `0..<maxElements`.
/// The native wrapper checks membership in C++ without calling back into Swift for each candidate.
public struct HNSWLabelBitset: Sendable {
    var words: [UInt64]
    public let capacity: Int

    /// Creates an empty bitset with one bit per possible label.
    public init(capacity: Int) {
        precondition(capacity >= 0, "capacity must be non-negative")
        self.capacity = capacity
        self.words = [UInt64](repeating: 0, count: (capacity + 63) / 64)
    }

    /// Marks a label as allowed.
    public mutating func insert(_ label: UInt64) {
        let index = self.requireValidLabel(label)
        self.words[index >> 6] |= 1 << UInt64(index & 63)
    }

    /// Marks a label as disallowed.
    public mutating func remove(_ label: UInt64) {
        let index = self.requireValidLabel(label)
        self.words[index >> 6] &= ~(1 << UInt64(index & 63))
    }

    /// Returns whether the label is allowed.
    public func contains(_ label: UInt64) -> Bool {
        guard let index = Int(exactly: label), index >= 0, index < self.capacity else {
            return false
        }
        return (self.words[index >> 6] & (1 << UInt64(index & 63))) != 0
    }

    /// Removes all allowed labels while preserving capacity.
    public mutating func removeAll() {
        self.words = [UInt64](repeating: 0, count: self.words.count)
    }

    private func requireValidLabel(_ label: UInt64) -> Int {
        guard let index = Int(exactly: label), index >= 0, index < self.capacity else {
            preconditionFailure("label must be within bitset capacity")
        }
        return index
    }
}

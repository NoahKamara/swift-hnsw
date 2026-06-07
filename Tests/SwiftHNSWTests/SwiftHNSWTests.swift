import Testing
@testable import SwiftHNSW

@Suite("HNSW Index Tests")
struct SwiftHNSWTests {

    @Test("Create index and add points")
    func testCreateIndexAndAddPoints() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        // Add some test vectors
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 1)
        try index.add([0.0, 0.0, 1.0, 0.0], label: 2)
        try index.add([0.0, 0.0, 0.0, 1.0], label: 3)

        #expect(index.count == 4)
    }

    @Test("Search for nearest neighbors")
    func testSearch() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        // Add test vectors
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.9, 0.1, 0.0, 0.0], label: 1)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 2)
        try index.add([0.0, 0.0, 1.0, 0.0], label: 3)

        // Search for nearest to [1, 0, 0, 0]
        let results = try index.search([1.0, 0.0, 0.0, 0.0], k: 2)

        #expect(results.count == 2)
        #expect(results[0].label == 0) // Exact match should be first
        #expect(results[0].distance == 0.0) // Distance should be 0
    }

    @Test("Dimension mismatch throws error")
    func testDimensionMismatch() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100
        )

        #expect(throws: HNSWError.self) {
            try index.add([1.0, 0.0, 0.0], label: 0) // Only 3 dimensions
        }
    }

    @Test("Inner product distance")
    func testInnerProductDistance() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100,
            metric: .innerProduct
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.5, 0.5, 0.0, 0.0], label: 1)

        let results = try index.search([1.0, 0.0, 0.0, 0.0], k: 2)
        #expect(results.count == 2)
    }

    @Test("Cosine similarity")
    func testCosineSimilarity() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100,
            metric: .cosine
        )

        // Different magnitudes but same direction
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([2.0, 0.0, 0.0, 0.0], label: 1)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 2)

        let results = try index.search([3.0, 0.0, 0.0, 0.0], k: 2)
        #expect(results.count == 2)
        // Labels 0 and 1 should be closest (same direction)
        let topLabels = Set(results.map { $0.label })
        #expect(topLabels.contains(0) || topLabels.contains(1))
    }

    @Test("Set ef search parameter")
    func testSetEfSearch() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 100,
            configuration: HNSWConfiguration(efSearch: 10)
        )

        index.setEfSearch(50)
        // Just verify it doesn't crash
        #expect(Bool(true))
    }

    @Test("Per-query ef improves recall")
    func testPerQueryEf() throws {
        let dimensions = 32
        let numElements = 500
        let k = 10
        let vectors = (0..<numElements).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }

        let index = try HNSWIndex<Float>(
            dimensions: dimensions,
            maxElements: numElements,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 10)
        )

        for (i, vector) in vectors.enumerated() {
            try index.add(vector, label: UInt64(i))
        }

        let queries = (0..<20).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
        let groundTruths = queries.map { bruteForceSearch(query: $0, vectors: vectors, k: k) }

        let lowEfResults = try queries.map { try index.search($0, k: k, ef: 10) }
        let highEfResults = try queries.map { try index.search($0, k: k, ef: 200) }

        let lowRecall = calculateAverageRecall(results: lowEfResults, groundTruths: groundTruths, k: k)
        let highRecall = calculateAverageRecall(results: highEfResults, groundTruths: groundTruths, k: k)

        #expect(highRecall >= lowRecall)
        #expect(highRecall > lowRecall || highRecall >= 0.9)
    }

    @Test("Replace deleted slot reuses element count")
    func testReplaceDeleted() throws {
        let config = HNSWConfiguration(allowReplaceDeleted: true)
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 10,
            metric: .l2,
            configuration: config
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        #expect(index.count == 1)

        try index.markDeleted(label: 0)
        #expect(index.contains(label: 0) == false)

        try index.add([0.0, 1.0, 0.0, 0.0], label: 1, replaceDeleted: true)
        #expect(index.count == 1)

        let results = try index.search([0.0, 1.0, 0.0, 0.0], k: 1)
        #expect(results.count == 1)
        #expect(results[0].label == 1)
        #expect(results[0].distance < 0.01)
    }

    @Test("Batch delete marks all labels and restores on unmark")
    func testBatchDelete() throws {
        let numElements = 32
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: numElements,
            metric: .l2
        )

        for i in 0..<numElements {
            var vector = [Float](repeating: 0, count: 4)
            vector[i % 4] = 1.0
            try index.add(vector, label: UInt64(i))
        }

        let labels = (0..<numElements).map(UInt64.init)
        try index.markDeleted(labels: labels)

        for label in labels {
            #expect(index.contains(label: label) == false)
        }

        let results = try index.search([1.0, 0.0, 0.0, 0.0], k: numElements)
        #expect(results.isEmpty)

        try index.unmarkDeleted(labels: labels)

        for label in labels {
            #expect(index.contains(label: label))
        }

        let restored = try index.search([1.0, 0.0, 0.0, 0.0], k: numElements)
        #expect(restored.count == numElements)
        #expect(Set(restored.map(\.label)) == Set(labels))
    }

    @Test("Batch delete matches single delete behavior")
    func testBatchDeleteMatchesSingleDelete() throws {
        let dimensions = 8
        let numElements = 16
        let singleIndex = try HNSWIndex<Float>(
            dimensions: dimensions,
            maxElements: numElements,
            metric: .l2
        )
        let batchIndex = try HNSWIndex<Float>(
            dimensions: dimensions,
            maxElements: numElements,
            metric: .l2
        )

        for i in 0..<numElements {
            var vector = [Float](repeating: 0, count: dimensions)
            vector[i % dimensions] = Float(i + 1)
            try singleIndex.add(vector, label: UInt64(i))
            try batchIndex.add(vector, label: UInt64(i))
        }

        let query = [Float](repeating: 0.25, count: dimensions)
        let labelsToDelete = [UInt64(1), 3, 5, 7]

        for label in labelsToDelete {
            try singleIndex.markDeleted(label: label)
        }
        try batchIndex.markDeleted(labels: labelsToDelete)

        let singleResults = try singleIndex.search(query, k: numElements)
        let batchResults = try batchIndex.search(query, k: numElements)

        #expect(Set(singleResults.map(\.label)) == Set(batchResults.map(\.label)))
        for label in labelsToDelete {
            #expect(singleIndex.contains(label: label) == false)
            #expect(batchIndex.contains(label: label) == false)
        }
    }

    @Test("Replace deleted without allowReplaceDeleted throws")
    func testReplaceDeletedNotEnabled() throws {
        let index = try HNSWIndex<Float>(
            dimensions: 4,
            maxElements: 10,
            metric: .l2
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.markDeleted(label: 0)

        #expect(throws: HNSWError.replaceDeletedNotEnabled) {
            try index.add([0.0, 1.0, 0.0, 0.0], label: 1, replaceDeleted: true)
        }
    }

    @Test("Large scale test")
    func testLargeScale() throws {
        let dimensions = 128
        let numElements = 1000
        let index = try HNSWIndex<Float>(
            dimensions: dimensions,
            maxElements: numElements,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        )

        // Generate random vectors
        for i in 0..<numElements {
            var vector = [Float](repeating: 0, count: dimensions)
            for j in 0..<dimensions {
                vector[j] = Float.random(in: -1...1)
            }
            try index.add(vector, label: UInt64(i))
        }

        #expect(index.count == numElements)

        // Search
        var query = [Float](repeating: 0, count: dimensions)
        for i in 0..<dimensions {
            query[i] = Float.random(in: -1...1)
        }

        let results = try index.search(query, k: 10)
        #expect(results.count == 10)
    }
}

// MARK: - Float16 Tests

@Suite("Float16 Index Tests")
struct Float16Tests {

    @Test("Create Float16 index and add points")
    func testCreateFloat16IndexAndAddPoints() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        // Add some test vectors
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 1)
        try index.add([0.0, 0.0, 1.0, 0.0], label: 2)
        try index.add([0.0, 0.0, 0.0, 1.0], label: 3)

        #expect(index.count == 4)
    }

    @Test("Float16 search for nearest neighbors")
    func testFloat16Search() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        // Add test vectors
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.9, 0.1, 0.0, 0.0], label: 1)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 2)
        try index.add([0.0, 0.0, 1.0, 0.0], label: 3)

        // Search for nearest to [1, 0, 0, 0]
        let query: [Float16] = [1.0, 0.0, 0.0, 0.0]
        let results = try index.search(query, k: 2)

        #expect(results.count == 2)
        #expect(results[0].label == 0) // Exact match should be first
    }

    @Test("Float16 inner product")
    func testFloat16InnerProduct() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .innerProduct
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.5, 0.5, 0.0, 0.0], label: 1)

        let query: [Float16] = [1.0, 0.0, 0.0, 0.0]
        let results = try index.search(query, k: 2)
        #expect(results.count == 2)
    }

    @Test("Float16 cosine similarity")
    func testFloat16Cosine() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .cosine
        )

        // Different magnitudes but same direction
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([2.0, 0.0, 0.0, 0.0], label: 1)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 2)

        let query: [Float16] = [3.0, 0.0, 0.0, 0.0]
        let results = try index.search(query, k: 2)
        #expect(results.count == 2)
    }

    @Test("Float16 large scale test")
    func testFloat16LargeScale() throws {
        let dimensions = 128
        let numElements = 1000
        let index = try HNSWIndex<Float16>(
            dimensions: dimensions,
            maxElements: numElements,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        )

        // Generate random vectors
        for i in 0..<numElements {
            let vector: [Float16] = (0..<dimensions).map { _ in Float16.random(in: -1...1) }
            try index.add(vector, label: UInt64(i))
        }

        #expect(index.count == numElements)

        // Search
        let query: [Float16] = (0..<dimensions).map { _ in Float16.random(in: -1...1) }
        let results = try index.search(query, k: 10)
        #expect(results.count == 10)
    }

    @Test("Float16 getVector roundtrip")
    func testFloat16GetVector() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        let vector: [Float16] = [1.0, 0.5, 0.25, 0.125]
        try index.add(vector, label: 42)

        let retrieved = index.getVector(label: 42)
        #expect(retrieved != nil)

        // Check values are approximately equal (Float16 has limited precision)
        if let retrieved = retrieved {
            for i in 0..<4 {
                let diff = abs(Float(retrieved[i]) - Float(vector[i]))
                #expect(diff < 0.01)
            }
        }
    }

    @Test("Float16 serialization roundtrip")
    func testFloat16Serialization() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 64,
            maxElements: 100,
            metric: .l2
        )

        // Add some vectors
        for i in 0..<50 {
            let vector: [Float16] = (0..<64).map { _ in Float16.random(in: -1...1) }
            try index.add(vector, label: UInt64(i))
        }

        // Serialize
        let data = try index.serialize()
        #expect(data.count > 0)

        // Load
        let loaded = try HNSWIndex<Float16>.load(
            from: data,
            dimensions: 64,
            metric: .l2
        )

        #expect(loaded.count == 50)
    }

    @Test("Type alias HNSWIndexF16 works")
    func testTypeAlias() throws {
        let index = try HNSWIndexF16(
            dimensions: 4,
            maxElements: 100,
            metric: .l2
        )

        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        #expect(index.count == 1)
    }
}

@Suite("Allowed-label filtered search")
struct AllowedLabelFilteredSearchTests {
    private func buildIndex(
        count: Int,
        dimensions: Int = 32
    ) throws -> HNSWIndex<Float> {
        let index = try HNSWIndex<Float>(
            dimensions: dimensions,
            maxElements: count,
            metric: .l2,
            configuration: HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 64)
        )

        for i in 0..<count {
            let vector = (0..<dimensions).map { dim in
                Float(i &+ dim &+ 1) * 0.01
            }
            try index.add(vector, label: UInt64(i))
        }

        return index
    }

    @Test("Bitset search returns only allowed labels")
    func testAllowedLabelsOnly() throws {
        let index = try buildIndex(count: 128)
        let query = (0..<32).map { Float($0 + 1) * 0.01 }
        var allowedLabels = HNSWLabelBitset(capacity: index.capacity)
        for label in stride(from: 0, to: 128, by: 2) {
            allowedLabels.insert(UInt64(label))
        }

        let results = try index.search(
            query,
            k: 10,
            allowedLabels: allowedLabels,
            ef: 256
        )

        #expect(!results.isEmpty)
        for result in results {
            #expect(allowedLabels.contains(result.label))
            #expect(Int(result.label).isMultiple(of: 2))
        }
    }

    @Test("Empty bitset returns no results")
    func testEmptyBitsetReturnsNoResults() throws {
        let index = try buildIndex(count: 32)
        let query = (0..<32).map { Float($0 + 1) * 0.01 }
        let allowedLabels = HNSWLabelBitset(capacity: index.capacity)

        let results = try index.search(
            query,
            k: 10,
            allowedLabels: allowedLabels,
            ef: 64
        )

        #expect(results.isEmpty)
    }

    @Test("High allowed label within capacity is matched")
    func testHighAllowedLabelWithinCapacity() throws {
        let index = try buildIndex(count: 130)
        let query = (0..<32).map { Float($0 + 1) * 0.01 }
        var allowedLabels = HNSWLabelBitset(capacity: index.capacity)
        allowedLabels.insert(129)

        let results = try index.search(
            query,
            k: 10,
            allowedLabels: allowedLabels,
            ef: 256
        )

        #expect(results.map(\.label) == [129])
    }

    @Test("Out-of-range labels are not considered allowed")
    func testOutOfRangeLabelsAreDisallowed() {
        var allowedLabels = HNSWLabelBitset(capacity: 4)
        allowedLabels.insert(3)

        #expect(allowedLabels.contains(3))
        #expect(!allowedLabels.contains(4))
        #expect(!allowedLabels.contains(UInt64.max))
    }

    @Test("Deleted labels are excluded from allowed-label search")
    func testDeletedLabelsExcluded() throws {
        let index = try buildIndex(count: 64)
        let query = (0..<32).map { Float($0 + 1) * 0.01 }
        let deletedLabel = UInt64(0)
        var allowedLabels = HNSWLabelBitset(capacity: index.capacity)
        for label in stride(from: 0, to: 64, by: 2) {
            allowedLabels.insert(UInt64(label))
        }

        try index.markDeleted(label: deletedLabel)
        #expect(index.contains(label: deletedLabel) == false)

        let results = try index.search(
            query,
            k: 10,
            allowedLabels: allowedLabels,
            ef: 256
        )

        #expect(!results.contains { $0.label == deletedLabel })
        for result in results {
            #expect(allowedLabels.contains(result.label))
        }
    }

    @Test("Float16 allowed-label search uses the bitset")
    func testFloat16AllowedLabelSearch() throws {
        let index = try HNSWIndex<Float16>(
            dimensions: 4,
            maxElements: 8,
            metric: .l2
        )
        try index.add([1.0, 0.0, 0.0, 0.0], label: 0)
        try index.add([0.0, 1.0, 0.0, 0.0], label: 1)

        var allowedLabels = HNSWLabelBitset(capacity: index.capacity)
        allowedLabels.insert(1)

        let results = try index.search(
            [1.0, 0.0, 0.0, 0.0],
            k: 2,
            allowedLabels: allowedLabels,
            ef: 16
        )

        #expect(results.map(\.label) == [1])
    }
}

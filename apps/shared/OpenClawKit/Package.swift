// swift-tools-version: 6.1

import Foundation
import PackageDescription

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

var packageTargets: [Target] = [
    .target(
        name: "OpenClawProtocol",
        path: "Sources/OpenClawProtocol",
        swiftSettings: [
            .enableUpcomingFeature("StrictConcurrency"),
        ]),
    .target(
        name: "OpenClawKit",
        dependencies: [
            "OpenClawProtocol",
        ],
        path: "Sources/OpenClawKit",
        exclude: [
            "ElevenLabsKitShim.swift",
            "AudioStreamingProtocols.swift",
            "TalkPromptBuilder.swift",
        ],
        resources: [
            .process("Resources"),
        ],
        swiftSettings: [
            .enableUpcomingFeature("StrictConcurrency"),
        ]),
    .target(
        name: "OpenClawChatUI",
        dependencies: [
            "OpenClawKit",
            .product(
                name: "Textual",
                package: "textual",
                condition: .when(platforms: [.macOS, .iOS])),
        ],
        path: "Sources/OpenClawChatUI",
        swiftSettings: [
            .enableUpcomingFeature("StrictConcurrency"),
        ]),
]

if FileManager.default.fileExists(
    atPath: packageRoot.appendingPathComponent("Tests/OpenClawKitTests").path)
{
    packageTargets.append(
        .testTarget(
            name: "OpenClawKitTests",
            dependencies: ["OpenClawKit", "OpenClawChatUI"],
            path: "Tests/OpenClawKitTests",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("SwiftTesting"),
            ]))
}

let package = Package(
    name: "OpenClawKit",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "OpenClawProtocol", targets: ["OpenClawProtocol"]),
        .library(name: "OpenClawKit", targets: ["OpenClawKit"]),
        .library(name: "OpenClawChatUI", targets: ["OpenClawChatUI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/gonzalezreal/textual", exact: "0.3.1"),
    ],
    targets: packageTargets)

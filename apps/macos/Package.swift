// swift-tools-version: 6.2
// Package manifest for the Marv macOS companion (menu bar app + IPC library).

import PackageDescription

let package = Package(
    name: "Marv",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "MarvIPC", targets: ["MarvIPC"]),
        .library(name: "MarvDiscovery", targets: ["MarvDiscovery"]),
        .executable(name: "Marv", targets: ["Marv"]),
        .executable(name: "marv-mac", targets: ["MarvMacCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/orchetect/MenuBarExtraAccess", exact: "1.2.2"),
        .package(url: "https://github.com/swiftlang/swift-subprocess.git", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.8.0"),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.8.1"),
        .package(url: "https://github.com/steipete/Peekaboo.git", branch: "main"),
        .package(path: "../shared/MarvKit"),
        .package(path: "../../Swabble"),
    ],
    targets: [
        .target(
            name: "MarvIPC",
            dependencies: [],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "MarvDiscovery",
            dependencies: [
                .product(name: "MarvKit", package: "MarvKit"),
            ],
            path: "Sources/MarvDiscovery",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .executableTarget(
            name: "Marv",
            dependencies: [
                "MarvIPC",
                "MarvDiscovery",
                .product(name: "MarvKit", package: "MarvKit"),
                .product(name: "MarvChatUI", package: "MarvKit"),
                .product(name: "MarvProtocol", package: "MarvKit"),
                .product(name: "SwabbleKit", package: "swabble"),
                .product(name: "MenuBarExtraAccess", package: "MenuBarExtraAccess"),
                .product(name: "Subprocess", package: "swift-subprocess"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Sparkle", package: "Sparkle"),
                .product(name: "PeekabooBridge", package: "Peekaboo"),
                .product(name: "PeekabooAutomationKit", package: "Peekaboo"),
            ],
            exclude: [
                "Resources/Info.plist",
            ],
            resources: [
                .copy("Resources/Marv.icns"),
                .copy("Resources/DeviceModels"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .executableTarget(
            name: "MarvMacCLI",
            dependencies: [
                "MarvDiscovery",
                .product(name: "MarvKit", package: "MarvKit"),
                .product(name: "MarvProtocol", package: "MarvKit"),
            ],
            path: "Sources/MarvMacCLI",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .testTarget(
            name: "MarvIPCTests",
            dependencies: [
                "MarvIPC",
                "Marv",
                "MarvDiscovery",
                .product(name: "MarvProtocol", package: "MarvKit"),
                .product(name: "SwabbleKit", package: "swabble"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("SwiftTesting"),
            ]),
    ])

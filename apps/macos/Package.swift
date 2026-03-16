// swift-tools-version: 6.1
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
        .package(path: "../shared/OpenClawKit"),
    ],
    targets: [
        .target(
            name: "MarvIPC",
            dependencies: [],
            path: "Sources/MarvIPC",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "MarvDiscovery",
            dependencies: [
                .product(name: "OpenClawKit", package: "OpenClawKit"),
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
                .product(name: "OpenClawKit", package: "OpenClawKit"),
                .product(name: "OpenClawChatUI", package: "OpenClawKit"),
                .product(name: "OpenClawProtocol", package: "OpenClawKit"),
                .product(name: "MenuBarExtraAccess", package: "MenuBarExtraAccess"),
                .product(name: "Subprocess", package: "swift-subprocess"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Sparkle", package: "Sparkle"),
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
                .product(name: "OpenClawKit", package: "OpenClawKit"),
                .product(name: "OpenClawProtocol", package: "OpenClawKit"),
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
                .product(name: "OpenClawProtocol", package: "OpenClawKit"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("SwiftTesting"),
            ]),
    ])

#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
build_root=$(mktemp -d /tmp/stereocapture-icon-test.XXXXXX)
trap 'rm -rf "$build_root"' EXIT

xcodebuild -quiet \
  -project "$repo_root/ios/StereoCapture.xcodeproj" \
  -scheme StereoCapture \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "$build_root/DerivedData" \
  CODE_SIGNING_ALLOWED=NO \
  build

app="$build_root/DerivedData/Build/Products/Debug-iphoneos/StereoCapture.app"
test -f "$app/Assets.car"
xcrun assetutil --info "$app/Assets.car" >"$build_root/assets.json"
grep -q '"AssetType"' "$build_root/assets.json"
icon_name=$(plutil -extract CFBundleIcons.CFBundlePrimaryIcon.CFBundleIconName raw "$app/Info.plist")
test "$icon_name" = "AppIcon"
echo "App icon validation passed"

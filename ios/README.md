# Stereo Capture iPhone setup

Stereo Capture records synchronized inputs for a rigid two-phone stereo rig:
RGB images, ARKit poses and intrinsics, audio, GPS, compass headings, and LiDAR
scene depth where the device supports it. You install the same Xcode project on
both phones and assign one phone **Camera A** and the other **Camera B**.

The phones do not need to be an iPhone 15 and iPhone 17 Pro. Those are only the
devices used during development. Use two ARKit-capable iPhones that can run
iOS 17 or newer. Once calibrated, keep the same phones, lenses, capture profile,
rig positions, and A/B assignments for the experiment.

## What you need

- A Mac running a compatible recent version of Xcode
- Two ARKit-capable iPhones running iOS 17 or newer
- One data-capable USB cable for initial pairing and installation
- An Apple Account signed into Xcode
- Free storage on both phones for RGB, audio, and sensor data
- The rigid two-phone mount used for calibration and capture

An App Store submission and paid Apple Developer membership are not required
for installing the app on your own phones. A free Xcode Personal Team is
sufficient, although its development signing normally expires after seven days.
Reconnect the phone and run the app from Xcode again when renewal is required.

## 1. Open the existing project

Clone the repository, then open:

```text
ios/StereoCapture.xcodeproj
```

From Xcode's welcome screen choose **Open Existing Project**. Do not create a
new project. If you cloned the repository through another Git client, navigate
to the repository and select `StereoCapture.xcodeproj`.

## 2. Configure signing

1. In Xcode, open **Xcode → Settings → Accounts** and sign in with your Apple Account.
2. Select the blue `StereoCapture` project in the Project navigator.
3. Select the `StereoCapture` target.
4. Open **Signing & Capabilities**.
5. Enable **Automatically manage signing**.
6. Select your Personal Team.
7. If Xcode reports that the bundle identifier is unavailable, replace
   `com.allison.multistereocapture` with a unique reverse-domain value such as
   `com.yourname.multistereocapture`.

The selected team is a local developer setting and should not be committed to
the public repository.

## 3. Pair the first phone

1. Connect the unlocked iPhone directly to the Mac with a data-capable cable.
2. Tap **Trust** on the phone if prompted and enter its passcode.
3. Enable **Developer Mode** under **Settings → Privacy & Security → Developer Mode**.
4. Restart the phone if iOS requests it, then confirm Developer Mode after restart.
5. Keep the phone unlocked while Xcode finishes preparing it.
6. At the top of Xcode, click the run destination currently showing something
   like **My Mac** or **Any iOS Device**.
7. Select the physical iPhone—not an iPhone simulator.

Recent Xcode releases use **Device Hub** instead of the older **Devices and
Simulators** window. Choose **Manage Devices…** at the bottom of the run
destination menu if you need to inspect the connection.

If the phone is absent from both Xcode and Finder, fix the USB connection first:
unlock it, reconnect it, try another port or cable, avoid an unpowered hub, and
accept the Trust prompt. Finder must recognize the phone before Xcode can use it.

## 4. Install Stereo Capture

With the physical phone selected as the run destination:

1. Press the Xcode **Run** button or `Command-R`.
2. Keep the phone unlocked while Xcode builds and installs the app.
3. Accept the camera, microphone, and location permission prompts.
4. Confirm that the live camera preview fills the screen.

If iOS asks you to trust the development certificate, open **Settings → General
→ VPN & Device Management**, select the Apple Account, and trust it. If Xcode
uses a stale build, choose **Product → Clean Build Folder** with `Shift-Command-K`
and run again.

Repeat the pairing and installation steps for the second phone. Both phones run
the same application; their A/B roles are selected inside the app.

## 5. Optional wireless Xcode connection

After a successful wired pairing and installation, Xcode may connect wirelessly
when the Mac and phone are on the same Wi-Fi network and Bluetooth is enabled on
both. Build and run over the cable once, unplug it, wait briefly, then check the
run-destination menu.

Some guest, campus, enterprise, VPN, hotspot, or client-isolated networks block
local device discovery. Wireless Xcode deployment is optional: install by cable,
then disconnect and record normally. The app does not need the Mac while recording.

## 6. Configure the two phones

Before recording, mount both phones rigidly and open Stereo Capture on each one.

Use matching settings:

- Enter exactly the same **session ID** on both phones.
- Assign one phone **Camera A** and the other **Camera B**.
- Select the same saved frame rate: 5, 10, or 15 FPS.
- Keep the same physical lenses and rig orientation used during calibration.

The recorder requests the same manual camera profile on both phones:

- 1/120-second exposure
- ISO 100
- 5000 K white balance
- lens position 0.75

The app records the values each phone actually achieves because different camera
modules may clamp or interpret the same request differently. ARKit continues to
track at its supported native video rate while the app saves RGB frames at the
selected lower rate.

### LiDAR behavior

LiDAR is capability-based, not permanently assigned to Camera A or Camera B.
Each phone checks whether ARKit scene depth is available:

- A LiDAR-capable phone records scene-depth and confidence buffers.
- A phone without LiDAR records the remaining streams normally.
- If both phones support LiDAR, both session exports contain LiDAR data.

The current offline stereo evaluator uses one LiDAR stream as the comparison
reference. With `--lidar-camera auto`, it chooses Camera A when A contains depth,
otherwise Camera B, otherwise no LiDAR. Override it with `--lidar-camera A`,
`--lidar-camera B`, or `--lidar-camera none`. It does not currently fuse both
LiDAR streams into a single reference depth map.

## 7. Record a paired session

1. Start recording on both phones as close together as practical.
2. At the beginning, place your hands in both cameras' shared field of view and
   perform three clearly separated claps.
3. Capture the calibration board or experimental route.
4. Before stopping, repeat the three visible and audible claps.
5. Stop and save on both phones.

The claps provide a shared audio event for offline clock alignment. Starting the
buttons simultaneously is helpful but is not the synchronization mechanism.

For extrinsic calibration, keep the board still for about one second at each
position. After extrinsic calibration, do not change the rigid relationship
between the two phones.

## 8. Export both sessions

After stopping, use **Export** on each phone and transfer the complete session
folders to the Mac using AirDrop, Files, or another file-transfer method. Keep
the folder contents intact; do not copy only the RGB images.

Each export can include:

```text
session/
├── manifest.json
├── frames.csv
├── audio.m4a
├── location.csv
├── heading.csv
├── rgb/
├── depth/          # LiDAR devices only
└── confidence/     # LiDAR devices only
```

Deleting or reinstalling the application may remove sessions that remain only
inside its app container. Export important recordings before reinstalling or
renewing the development build.

## 9. Validate and import on the Mac

From the repository root with the Python environment activated:

```bash
python -m calibration import-session /path/to/session_A /path/to/session_B
```

The importer verifies that the session IDs match and the roles are A and B. It
writes validated timestamps and reports under `calibration/outputs/`. Continue
with synchronization and calibration using the commands in the main
[README](../README.md).

## Troubleshooting

### The phone disappears when the cable is removed

Wireless deployment was not established or the network blocks device discovery.
Reconnect by cable to install or debug. Once installed, Stereo Capture itself
runs without either a cable or wireless Xcode connection.

### Xcode shows the phone as unavailable

Unlock the phone, verify Developer Mode, reconnect the cable, and check whether
Finder lists it under Locations. Quit and reopen Xcode after Finder recognizes it.

### Installation reports an invalid bundle or signing error

Pull the latest repository changes, confirm a Personal Team is selected, clean
the build folder, and run again. Use a unique bundle identifier if Xcode reports
a registration conflict.

### The app stops opening after several days

Free Personal Team provisioning expires periodically. Reconnect the phone,
select it as the Xcode run destination, and press `Command-R` to renew the build.

### Only one phone contains depth files

That is expected when only one device supports LiDAR. Keep `--lidar-camera auto`
or explicitly select the LiDAR phone's A/B role during offline processing.

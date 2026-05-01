# Chiranjeevi Flutter App — Phase 1 Implementation Plan
## Project Scaffolding, M3 Expressive Theming & App Shell

---

## Context & Scope

**Flutter SDK:** 3.41.0 (stable, installed) · **Dart:** 3.11.0  
**App name:** `chiranjeevi` · **Org:** `com.chiranjeevi`  
**Location:** `d:\techgium\mobile\`  
**Min SDK:** Android 21 (API 21, Android 5.0)  

Phase 1 delivers **everything except actual scanning/downloading** — it is a fully navigable, themed app shell. By end of Phase 1 you can run the app on an emulator/device, see the home screen, navigate between screens via drawer and bottom nav, and see all empty-state screens. No real data flows yet.

---

## What Gets Delivered in Phase 1

```
mobile/
├── android/
│   └── app/
│       └── build.gradle          ← minSdkVersion 21, compileSdk 35
├── lib/
│   ├── main.dart                 ← Entry point, ProviderScope
│   ├── app.dart                  ← MaterialApp.router + routing
│   ├── theme/
│   │   └── app_theme.dart        ← Full M3 Expressive theme (light + dark)
│   ├── models/
│   │   └── report_metadata.dart  ← Data class (no DB yet)
│   ├── screens/
│   │   ├── home_screen.dart      ← Main scaffold, nav drawer, bottom nav, FAB
│   │   ├── scan_screen.dart      ← Placeholder scanner screen
│   │   └── history_screen.dart   ← Placeholder history screen
│   └── widgets/
│       └── empty_state.dart      ← Reusable empty state widget
├── pubspec.yaml                  ← All 9 dependencies pinned
└── README.md
```

> [!IMPORTANT]
> Phase 1 only adds `provider` and `google_fonts` + `flex_color_scheme` as working dependencies. All other packages (`mobile_scanner`, `dio`, `sqflite`, etc.) are added to `pubspec.yaml` now but are **not wired up** until Phase 2. This ensures `flutter pub get` succeeds and the project compiles cleanly.

---

## Proposed Changes — File by File

---

### Step 0 — Project Creation (Terminal Command)

```powershell
flutter create `
  --org com.chiranjeevi `
  --project-name chiranjeevi `
  --platforms android,ios `
  --empty `
  d:\techgium\mobile
```

**Flags explained:**
- `--org com.chiranjeevi` → Android package `com.chiranjeevi.chiranjeevi`
- `--project-name chiranjeevi` → Dart project name, overrides directory name
- `--platforms android,ios` → Skip web/desktop bloat
- `--empty` → Minimal `main.dart` with no demo counter boilerplate

---

### Step 1 — `pubspec.yaml`

#### [MODIFY] `mobile/pubspec.yaml`

Replace the scaffold-generated dependencies section with the full dependency list. **All packages pinned to latest stable:**

```yaml
name: chiranjeevi
description: "Chiranjeevi health report companion app — scan QR codes to access patient reports."
publish_to: 'none'
version: 1.0.0+1

environment:
  sdk: '>=3.11.0 <4.0.0'
  flutter: ">=3.41.0"

dependencies:
  flutter:
    sdk: flutter

  # QR Scanning
  mobile_scanner: ^7.2.0

  # HTTP Downloads
  dio: ^5.8.0

  # PDF Viewing
  pdfrx: ^1.0.0

  # Local Database (report metadata)
  sqflite: ^2.4.0

  # File path resolution
  path_provider: ^2.1.4

  # State Management
  provider: ^6.1.2

  # Camera/Storage Permissions
  permission_handler: ^11.4.0

  # Date formatting
  intl: ^0.20.2

  # Theming
  flex_color_scheme: ^8.4.0
  google_fonts: ^6.2.1

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^5.0.0

flutter:
  uses-material-design: true
```

---

### Step 2 — Android `build.gradle` Patch

#### [MODIFY] `mobile/android/app/build.gradle`

Change the `minSdk` from the Flutter default (flutter.minSdkVersion) to an explicit `21`:

```groovy
android {
    namespace "com.chiranjeevi.chiranjeevi"
    compileSdk 35

    defaultConfig {
        applicationId "com.chiranjeevi.chiranjeevi"
        minSdk 21          // ← was: flutter.minSdkVersion
        targetSdk 35
        versionCode flutterVersionCode.toInteger()
        versionName flutterVersionName
    }
    ...
}
```

Also add the **camera permission** and **internet permission** to `mobile/android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>

<uses-feature android:name="android.hardware.camera"/>
<uses-feature android:name="android.hardware.camera.autofocus"/>
```

---

### Step 3 — Theme System

#### [NEW] `mobile/lib/theme/app_theme.dart`

This is the **most important file in Phase 1**. Full M3 Expressive theme using `FlexColorScheme` + `GoogleFonts`.

**Color Strategy — Medical Teal Palette:**
- Seed: `Color(0xFF00897B)` (teal 600 — clinical, trusted, modern)
- Secondary: `Color(0xFF0288D1)` (light blue — tech/digital feel)
- Error: `Color(0xFFB00020)` (Material default red)

**Typography — Outfit font:**
- `displayLarge`: weight 800, tracking -0.5
- `headlineLarge`: weight 700
- `titleLarge`: weight 600
- `bodyMedium`: weight 400

**Component Themes:**
| Component | Override |
|---|---|
| `AppBarTheme` | Centered title, scrolled-under elevation with surface tint |
| `NavigationBarThemeData` | Height 72px, indicator shape Rounded (28px), label always visible |
| `FloatingActionButtonThemeData` | Large size, extended with icon + label |
| `CardTheme` | Border radius 20px, elevation 0 with outline, filled style |
| `NavigationDrawerThemeData` | Width 300px, indicator rounded |
| `ChipTheme` | StadiumBorder |

**Motion — Spring physics curves:**
```dart
static const springCurve = Curves.easeOutCubic;  // Page transitions
// Custom spring for FAB: uses TweenAnimationBuilder with springSimulation
```

**Full code:**

```dart
// mobile/lib/theme/app_theme.dart
import 'package:flex_color_scheme/flex_color_scheme.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  AppTheme._();

  static const Color _seedColor = Color(0xFF00897B);
  static const Color _secondaryColor = Color(0xFF0288D1);

  // ─── Light Theme ───────────────────────────────────────────────────────────
  static ThemeData get light {
    final flexScheme = FlexColorScheme.light(
      colors: FlexSchemeColor.from(
        primary: _seedColor,
        secondary: _secondaryColor,
      ),
      usedColors: 6,
      surfaceMode: FlexSurfaceMode.levelSurfacesLowScaffold,
      blendLevel: 9,
      subThemesData: _subThemesData(),
      visualDensity: FlexColorScheme.comfortablePlatformDensity,
      useMaterial3: true,
      useExpressiveOnContainerColors: true,
      textTheme: _buildTextTheme(brightness: Brightness.light),
    );
    return flexScheme.toTheme;
  }

  // ─── Dark Theme ────────────────────────────────────────────────────────────
  static ThemeData get dark {
    final flexScheme = FlexColorScheme.dark(
      colors: FlexSchemeColor.from(
        primary: _seedColor,
        secondary: _secondaryColor,
      ),
      usedColors: 6,
      surfaceMode: FlexSurfaceMode.levelSurfacesLowScaffold,
      blendLevel: 15,
      subThemesData: _subThemesData(),
      visualDensity: FlexColorScheme.comfortablePlatformDensity,
      useMaterial3: true,
      useExpressiveOnContainerColors: true,
      textTheme: _buildTextTheme(brightness: Brightness.dark),
    );
    return flexScheme.toTheme;
  }

  // ─── Shared Sub-Themes ─────────────────────────────────────────────────────
  static FlexSubThemesData _subThemesData() {
    return const FlexSubThemesData(
      // AppBar
      appBarScrolledUnderElevation: 4.0,
      appBarBackgroundSchemeColor: SchemeColor.surface,

      // Navigation Bar
      navigationBarHeight: 72,
      navigationBarSelectedLabelSchemeColor: SchemeColor.primary,
      navigationBarIndicatorSchemeColor: SchemeColor.primaryContainer,
      navigationBarIndicatorBorderRadius: 28.0,
      navigationBarLabelBehavior: NavigationDestinationLabelBehavior.alwaysShow,

      // Navigation Drawer
      drawerWidth: 300,
      drawerIndicatorBorderRadius: 24.0,
      drawerSelectedItemSchemeColor: SchemeColor.primary,

      // FAB
      fabUseShape: true,
      fabBorderRadius: 20.0,
      fabSchemeColor: SchemeColor.primary,

      // Cards
      cardBorderRadius: 20.0,
      cardElevation: 0.0,

      // Chips
      chipBorderRadius: 50.0,

      // Dialogs
      dialogBorderRadius: 28.0,

      // Buttons
      elevatedButtonBorderRadius: 28.0,
      filledButtonBorderRadius: 28.0,
      outlinedButtonBorderRadius: 28.0,
      textButtonBorderRadius: 28.0,
    );
  }

  // ─── Typography ────────────────────────────────────────────────────────────
  static TextTheme _buildTextTheme({required Brightness brightness}) {
    final base = GoogleFonts.outfitTextTheme();
    return base.copyWith(
      displayLarge: base.displayLarge?.copyWith(
        fontWeight: FontWeight.w800,
        letterSpacing: -0.5,
      ),
      displayMedium: base.displayMedium?.copyWith(
        fontWeight: FontWeight.w700,
      ),
      headlineLarge: base.headlineLarge?.copyWith(
        fontWeight: FontWeight.w700,
      ),
      headlineMedium: base.headlineMedium?.copyWith(
        fontWeight: FontWeight.w600,
      ),
      titleLarge: base.titleLarge?.copyWith(
        fontWeight: FontWeight.w600,
        letterSpacing: 0.15,
      ),
      titleMedium: base.titleMedium?.copyWith(
        fontWeight: FontWeight.w500,
      ),
      labelLarge: base.labelLarge?.copyWith(
        fontWeight: FontWeight.w600,
        letterSpacing: 0.5,
      ),
    );
  }
}
```

---

### Step 4 — Data Model (Shell Only)

#### [NEW] `mobile/lib/models/report_metadata.dart`

The model is created now so it can be referenced in the UI shell. No DB wiring yet.

```dart
class ReportMetadata {
  final int? id;
  final String reportId;
  final String? patientName;
  final DateTime downloadedAt;
  final String filePath;
  final int? fileSizeBytes;
  final String? sourceUrl;

  const ReportMetadata({
    this.id,
    required this.reportId,
    this.patientName,
    required this.downloadedAt,
    required this.filePath,
    this.fileSizeBytes,
    this.sourceUrl,
  });

  Map<String, dynamic> toMap() => {
    'report_id': reportId,
    'patient_name': patientName,
    'downloaded_at': downloadedAt.toIso8601String(),
    'file_path': filePath,
    'file_size_bytes': fileSizeBytes,
    'source_url': sourceUrl,
  };

  factory ReportMetadata.fromMap(Map<String, dynamic> map) => ReportMetadata(
    id: map['id'] as int?,
    reportId: map['report_id'] as String,
    patientName: map['patient_name'] as String?,
    downloadedAt: DateTime.parse(map['downloaded_at'] as String),
    filePath: map['file_path'] as String,
    fileSizeBytes: map['file_size_bytes'] as int?,
    sourceUrl: map['source_url'] as String?,
  );
}
```

---

### Step 5 — App Root

#### [NEW] `mobile/lib/app.dart`

Root `MaterialApp` with named routes and theme injection:

```dart
import 'package:flutter/material.dart';
import 'theme/app_theme.dart';
import 'screens/home_screen.dart';
import 'screens/scan_screen.dart';
import 'screens/history_screen.dart';

class ChiranjeeviApp extends StatelessWidget {
  const ChiranjeeviApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Chiranjeevi',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      darkTheme: AppTheme.dark,
      themeMode: ThemeMode.system,
      initialRoute: '/home',
      routes: {
        '/home': (_) => const HomeScreen(),
        '/scan': (_) => const ScanScreen(),
        '/history': (_) => const HistoryScreen(),
      },
      onGenerateRoute: (settings) {
        // Spring-physics page transition
        return PageRouteBuilder(
          settings: settings,
          pageBuilder: (context, animation, secondaryAnimation) {
            final child = _routeWidgetFor(settings.name);
            return child;
          },
          transitionDuration: const Duration(milliseconds: 400),
          reverseTransitionDuration: const Duration(milliseconds: 350),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            final tween = Tween(begin: const Offset(0.0, 0.04), end: Offset.zero)
                .chain(CurveTween(curve: Curves.easeOutCubic));
            final fadeTween = Tween<double>(begin: 0.0, end: 1.0)
                .chain(CurveTween(curve: Curves.easeOut));
            return FadeTransition(
              opacity: animation.drive(fadeTween),
              child: SlideTransition(
                position: animation.drive(tween),
                child: child,
              ),
            );
          },
        );
      },
    );
  }

  Widget _routeWidgetFor(String? name) {
    return switch (name) {
      '/scan'    => const ScanScreen(),
      '/history' => const HistoryScreen(),
      _          => const HomeScreen(),
    };
  }
}
```

#### [MODIFY] `mobile/lib/main.dart`

```dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Lock to portrait for the medical context
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  runApp(const ChiranjeeviApp());
}
```

---

### Step 6 — Home Screen (The Star of Phase 1)

#### [NEW] `mobile/lib/screens/home_screen.dart`

The main scaffold with:
- **Navigation Drawer** (left side swipe or hamburger)
- **Bottom Navigation Bar** with 3 items — Home | [FAB gap] | History
- **Center-docked QR FAB** (Google Pay style — prominent, branded)
- **Home Body** — branded welcome card + animated stats placeholder

**Layout decisions:**
- `BottomAppBar` with `notchMargin: 8.0` + `FloatingActionButtonLocation.centerDocked` for the Google Pay style
- Drawer header uses gradient with app logo/icon and app name
- Body contains a hero banner card ("Ready to scan") + an empty recent reports section

```
┌────────────────────────────────────┐
│  ☰  Chiranjeevi               ⚙   │  ← AppBar
├────────────────────────────────────┤
│                                    │
│  ┌────────────────────────────┐    │
│  │  🏥  Ready to scan         │    │
│  │  Point camera at QR code   │    │
│  │  to get patient report     │    │  ← Hero card
│  └────────────────────────────┘    │
│                                    │
│  Recent Reports                    │
│  ┌─────────────────────────────┐   │
│  │  [Empty state illustration] │   │  ← Empty state widget
│  └─────────────────────────────┘   │
│                                    │
├──────────┬──────────┬──────────────┤
│ 🏠 Home  │   [╋]    │  📋 History │  ← Bottom Nav + FAB
└──────────┴──────────┴──────────────┘
```

**Key code patterns:**

```dart
// Scaffold configuration
Scaffold(
  drawer: const _AppDrawer(),
  appBar: AppBar(
    title: const Text('Chiranjeevi'),
    centerTitle: false,
    actions: [/* settings icon */],
  ),
  body: const _HomeBody(),
  floatingActionButton: FloatingActionButton.large(
    heroTag: 'qr-scan-fab',
    onPressed: () => Navigator.pushNamed(context, '/scan'),
    tooltip: 'Scan QR Code',
    child: const Icon(Icons.qr_code_scanner_rounded, size: 36),
  ),
  floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
  bottomNavigationBar: BottomAppBar(
    shape: const CircularNotchedRectangle(),
    notchMargin: 10.0,
    height: 72,
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceAround,
      children: [
        // Left: Home
        IconButton(
          icon: const Icon(Icons.home_rounded),
          onPressed: () {},
          tooltip: 'Home',
        ),
        // Right gap for FAB
        const SizedBox(width: 72),
        // Right: History
        IconButton(
          icon: const Icon(Icons.history_rounded),
          onPressed: () => Navigator.pushNamed(context, '/history'),
          tooltip: 'History',
        ),
      ],
    ),
  ),
),
```

**Drawer content:**

```dart
class _AppDrawer extends StatelessWidget {
  const _AppDrawer();

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return NavigationDrawer(
      children: [
        // Gradient header
        DrawerHeader(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                colorScheme.primary,
                colorScheme.secondary,
              ],
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.health_and_safety_rounded, 
                   size: 48, color: colorScheme.onPrimary),
              const SizedBox(height: 12),
              Text('Chiranjeevi',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: colorScheme.onPrimary,
                  fontWeight: FontWeight.w800,
                )),
              Text('Health Report Companion',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: colorScheme.onPrimary.withOpacity(0.8),
                )),
            ],
          ),
        ),
        NavigationDrawerDestination(
          icon: const Icon(Icons.qr_code_scanner_rounded),
          label: const Text('Scan QR Code'),
        ),
        NavigationDrawerDestination(
          icon: const Icon(Icons.history_rounded),
          label: const Text('Report History'),
        ),
        const Divider(),
        // Future slots (greyed out)
        ListTile(
          leading: const Icon(Icons.settings_outlined),
          title: const Text('Settings'),
          enabled: false,
          subtitle: const Text('Coming soon'),
        ),
        ListTile(
          leading: const Icon(Icons.info_outline_rounded),
          title: const Text('About'),
          onTap: () { /* Phase 4 */ },
        ),
      ],
    );
  }
}
```

**Home body — Hero card + animated pulse on FAB:**

The hero card will have a subtle gradient and animated icon (pulse animation using `AnimationController` + `ScaleTransition`).

---

### Step 7 — Placeholder Screens

#### [NEW] `mobile/lib/screens/scan_screen.dart` (Phase 1 placeholder)

A proper-looking placeholder that will be replaced in Phase 2:

```dart
// Shows a dark screen with a scan frame illustration and "Camera coming in Phase 2" text
// Uses the same theme — no raw colors
Scaffold(
  appBar: AppBar(title: const Text('Scan QR')),
  backgroundColor: Colors.black87,
  body: Center(
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.qr_code_scanner_rounded, 
             size: 120, color: colorScheme.primary),
        const SizedBox(height: 24),
        Text('QR Scanner', style: textTheme.headlineMedium),
        Text('Camera integration in Phase 2'),
        // Animated scan line (CSS-style) using AnimatedContainer
      ],
    ),
  ),
)
```

#### [NEW] `mobile/lib/screens/history_screen.dart` (Phase 1 placeholder)

Uses the `EmptyState` widget:

```dart
Scaffold(
  appBar: AppBar(title: const Text('Report History')),
  body: const EmptyState(
    icon: Icons.folder_open_rounded,
    title: 'No Reports Yet',
    subtitle: 'Scan a QR code to download\nyour first health report',
    actionLabel: 'Scan Now',
    // actionOnPressed: navigate to scan
  ),
)
```

---

### Step 8 — Reusable Empty State Widget

#### [NEW] `mobile/lib/widgets/empty_state.dart`

```dart
class EmptyState extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final String? actionLabel;
  final VoidCallback? actionOnPressed;

  const EmptyState({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    this.actionLabel,
    this.actionOnPressed,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Animated icon container
            TweenAnimationBuilder<double>(
              tween: Tween(begin: 0.8, end: 1.0),
              duration: const Duration(milliseconds: 600),
              curve: Curves.elasticOut,
              builder: (context, value, child) =>
                  Transform.scale(scale: value, child: child),
              child: Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: colorScheme.primaryContainer,
                  shape: BoxShape.circle,
                ),
                child: Icon(icon, size: 56, color: colorScheme.onPrimaryContainer),
              ),
            ),
            const SizedBox(height: 32),
            Text(title, style: textTheme.headlineMedium, textAlign: TextAlign.center),
            const SizedBox(height: 12),
            Text(subtitle,
              style: textTheme.bodyLarge?.copyWith(color: colorScheme.onSurfaceVariant),
              textAlign: TextAlign.center,
            ),
            if (actionLabel != null) ...[
              const SizedBox(height: 32),
              FilledButton.icon(
                onPressed: actionOnPressed,
                icon: const Icon(Icons.qr_code_scanner_rounded),
                label: Text(actionLabel!),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
```

---

## Home Screen — Hero Banner Card Detail

The home screen body will have two sections:

### 1. Hero Banner Card
- Full-width `Card` with gradient overlay
- Animated pulse icon (health/QR icon)
- CTA button "Tap the button below to scan"
- Uses `TweenAnimationBuilder` for the entrance animation (fade + slide up)

### 2. Recent Reports Preview
- Section header "Recent Reports"
- Shows `EmptyState` widget (no reports yet)
- Will be populated in Phase 3

---

## Execution Order (Commands)

```powershell
# 1. Create the project
flutter create --org com.chiranjeevi --project-name chiranjeevi --platforms android,ios --empty d:\techgium\mobile

# 2. Get dependencies
cd d:\techgium\mobile
flutter pub get

# 3. Verify compilation
flutter analyze

# 4. Run on emulator/device
flutter run
```

---

## What Phase 1 Does NOT Include

| Excluded | Reason |
|---|---|
| QR camera | Phase 2 |
| PDF download logic | Phase 2 |
| SQLite DB wiring | Phase 2 |
| History list with real data | Phase 3 |
| PDF viewer | Phase 3 |
| Delete/download dialogs | Phase 3 |
| Animations beyond entrance | Phase 4 |
| Unit tests | Phase 4 |

---

## Verification Plan

After Phase 1 execution, I will verify:

1. **`flutter analyze`** → zero errors, zero warnings
2. **Hot reload** works without errors
3. **Light + dark theme** → toggle system dark mode, verify colors flip correctly
4. **Navigation** → Drawer opens, both drawer items navigate correctly; bottom nav History icon navigates correctly
5. **FAB** → tapping opens scan placeholder screen
6. **Typography** → Outfit font loads from Google Fonts (not fallback)
7. **Empty state animation** → elastic pop-in on first render

---

> [!NOTE]
> Phase 2 will be planned and started **only after you explicitly tell me to proceed**. Phase 1 is self-contained and delivers a fully runnable, well-themed app shell.

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
      navigationBarLabelBehavior: NavigationDestinationLabelBehavior.alwaysShow,

      // Navigation Drawer
      drawerWidth: 300,
      drawerSelectedItemSchemeColor: SchemeColor.primary,

      // FAB
      fabUseShape: true,
      fabSchemeColor: SchemeColor.primary,

      // Cards
      cardElevation: 0.0,

      defaultRadius: 28.0, // Used for dialogs, buttons, navigation indicators, etc
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

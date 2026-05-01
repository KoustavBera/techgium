import 'package:flutter/material.dart';

class ScanOverlay extends StatefulWidget {
  final bool isScanning;
  
  const ScanOverlay({super.key, this.isScanning = false});

  @override
  State<ScanOverlay> createState() => _ScanOverlayState();
}

class _ScanOverlayState extends State<ScanOverlay> with TickerProviderStateMixin {
  late final AnimationController _pulseController;
  late final AnimationController _scanLineController;
  late final Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _pulseAnimation = Tween<double>(begin: 0.97, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _scanLineController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _scanLineController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return Stack(
      children: [
        // Dark mask with transparent hole
        CustomPaint(
          size: Size.infinite,
          painter: _MaskPainter(),
        ),
        
        // Scan box with animated brackets and scan line
        Center(
          child: SizedBox(
            width: 280,
            height: 280,
            child: ScaleTransition(
              scale: _pulseAnimation,
              child: Stack(
                children: [
                  // Top-left bracket
                  Positioned(
                    top: 0,
                    left: 0,
                    child: _buildBracket(colorScheme.primary, top: true, left: true),
                  ),
                  // Top-right bracket
                  Positioned(
                    top: 0,
                    right: 0,
                    child: _buildBracket(colorScheme.primary, top: true, left: false),
                  ),
                  // Bottom-left bracket
                  Positioned(
                    bottom: 0,
                    left: 0,
                    child: _buildBracket(colorScheme.primary, top: false, left: true),
                  ),
                  // Bottom-right bracket
                  Positioned(
                    bottom: 0,
                    right: 0,
                    child: _buildBracket(colorScheme.primary, top: false, left: false),
                  ),
                  // Animated scan line
                  AnimatedBuilder(
                    animation: _scanLineController,
                    builder: (context, child) {
                      return Positioned(
                        top: _scanLineController.value * 276, // 280 - 4
                        left: 10,
                        right: 10,
                        child: Container(
                          height: 3,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                colorScheme.primary.withValues(alpha: 0.0),
                                colorScheme.primary,
                                colorScheme.primary.withValues(alpha: 0.0),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ),

        // Status text
        Positioned(
          bottom: 100,
          left: 0,
          right: 0,
          child: Center(
            child: Text(
              widget.isScanning ? 'Processing...' : 'Point at the QR code',
              style: textTheme.titleMedium?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildBracket(Color color, {required bool top, required bool left}) {
    const double length = 40.0;
    const double thickness = 4.0;
    const double radius = 12.0;

    return Container(
      width: length,
      height: length,
      decoration: BoxDecoration(
        border: Border(
          top: top ? BorderSide(color: color, width: thickness) : BorderSide.none,
          bottom: !top ? BorderSide(color: color, width: thickness) : BorderSide.none,
          left: left ? BorderSide(color: color, width: thickness) : BorderSide.none,
          right: !left ? BorderSide(color: color, width: thickness) : BorderSide.none,
        ),
        borderRadius: BorderRadius.only(
          topLeft: top && left ? const Radius.circular(radius) : Radius.zero,
          topRight: top && !left ? const Radius.circular(radius) : Radius.zero,
          bottomLeft: !top && left ? const Radius.circular(radius) : Radius.zero,
          bottomRight: !top && !left ? const Radius.circular(radius) : Radius.zero,
        ),
      ),
    );
  }
}

class _MaskPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.black.withValues(alpha: 0.65);
    final backgroundPath = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    
    // Create a 280x280 rounded rect in the center
    final rectSide = 280.0;
    final left = (size.width - rectSide) / 2;
    final top = (size.height - rectSide) / 2;
    final holePath = Path()
      ..addRRect(RRect.fromRectAndRadius(
        Rect.fromLTWH(left, top, rectSide, rectSide),
        const Radius.circular(16.0),
      ));
    
    // Combine paths (difference) to create the mask with a hole
    final maskPath = Path.combine(PathOperation.difference, backgroundPath, holePath);
    canvas.drawPath(maskPath, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

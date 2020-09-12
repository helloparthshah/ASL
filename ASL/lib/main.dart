import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as imglib;

List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(App());
}

class App extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Title',
      home: CameraApp(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class CameraApp extends StatefulWidget {
  @override
  _CameraAppState createState() => _CameraAppState();
}

class _CameraAppState extends State<CameraApp> {
  bool isDetecting = false;
  bool modelLoaded = false;
  CameraController controller;

  Future<void> loadModel() async {
    String res = await Tflite.loadModel(
      model: "assets/test.tflite",
    );
    print(res);
    setState(() {
      modelLoaded = true;
    });
  }

  @override
  void initState() {
    super.initState();
    loadModel();
    controller = CameraController(cameras[0], ResolutionPreset.ultraHigh,
        enableAudio: false);
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }

      controller.startImageStream((CameraImage image) {
        if (!modelLoaded) return;
        if (isDetecting) return;
        setState(() {
          isDetecting = true;
        });
        try {
          predict(image);
        } catch (e) {
          print(e);
        }
      });
    });
  }

  Future<Image> convertYUV420toImage(CameraImage image) async {
    try {
      final int width = image.width;
      final int height = image.height;

      // imgLib -> Image package from https://pub.dartlang.org/packages/image
      var img = imglib.Image(width, height); // Create Image buffer

      // Fill image buffer with plane[0] from YUV420_888
      for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
          final pixelColor = image.planes[0].bytes[y * width + x];
          // color: 0x FF  FF  FF  FF
          //           A   B   G   R
          // Calculate pixel color
          img.data[y * width + x] = (0xFF << 24) |
              (pixelColor << 16) |
              (pixelColor << 8) |
              pixelColor;
        }
      }

      imglib.PngEncoder pngEncoder = new imglib.PngEncoder(level: 0, filter: 0);
      List<int> png = pngEncoder.encodeImage(img);
      // muteYUVProcessing = false;
      return Image.memory(png);
    } catch (e) {
      print(">>>>>>>>>>>> ERROR:" + e.toString());
    }
    return null;
  }

  void predict(CameraImage image) async {
    // final i = await convertYUV420toImage(image);
    print(image.height.toString() + " " + image.width.toString());
    // final resize = imglib.copyResize(i, height: 96, width: 96);
    /* final i = convertYUV420toImageColor(image);
    im.copyResizeCropSquare(i, 96).getBytes(format: Format.rgb); */

    /* var pred = Tflite.runModelOnFrame(
        bytesList: image.planes.map((plane) {
      return plane.bytes;
    }).toList()); */

    /* print(image.planes.map((plane) {
      return plane.bytes;
    }).toList()); */
    setState(() {
      isDetecting = false;
    });
  }

  @override
  void dispose() {
    controller.dispose();
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return Container();
    }
    final size = MediaQuery.of(context).size;
    final deviceRatio = size.width / size.height;
    return Scaffold(
      body: Transform.scale(
        scale: controller.value.aspectRatio / deviceRatio,
        child: Center(
          child: AspectRatio(
            aspectRatio: controller.value.aspectRatio,
            child: CameraPreview(controller),
          ),
        ),
      ),
    );
  }
  /* return AspectRatio(
        aspectRatio: controller.value.aspectRatio,
        child: CameraPreview(controller));
  } */
}

#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"


int main(int argc, char* argv[])
{
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;
    const char windowName[] = "result";
    constexpr double avgalpha = 0.1;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    cmd.add<std::string>("image", 'i', "Image source to be detected.", false);
    cmd.add<std::string>("video", 'v', "Video/webcam source to be detected.", false);
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add("gpu", '\0', "Inference on cuda device.");
    cmd.add("fullscreen", '\0', "Display fullscreen.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    bool isFullScreen = cmd.exist("fullscreen");
    bool isImage = cmd.exist("image");
    bool isVideo = cmd.exist("video");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imagePath = cmd.get<std::string>("image");
    const std::string videoPath = cmd.get<std::string>("video");
    const std::string modelPath = cmd.get<std::string>("model_path");

    double avgduration = 0;
    bool windowCreated = false;

    if(!isImage && !isVideo) {
        std::cerr << "Either the --image or the --video argument has to be specified." << std::endl;
        std::cerr << std::endl;
        std::cerr << cmd.usage();
        return -1;
    }

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    YOLODetector detector {nullptr};
    try {
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    std::cout << "Model was initialized." << std::endl;

    cv::VideoCapture cap;
    if(isVideo) {
        try {
            cap = cv::VideoCapture(std::stoi(videoPath));
        } catch (std::logic_error &e) {
            cap = cv::VideoCapture(videoPath);
        }
    }

    do {
        auto start = std::chrono::steady_clock::now();

        cv::Mat image;
        std::vector<Detection> result;

        try {
            if(isImage) {
                image = cv::imread(imagePath);
            } else {
                cap >> image;
            }
        } catch(cv::Exception ex) {
            std::cout << ex.what() << std::endl;
        } catch(...) {
            std::cout << "Unknown exception" << std::endl;
        }

        try {
            result = detector.detect(image, confThreshold, iouThreshold);
        } catch(const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return -1;
        }

        utils::visualizeDetection(image, result, classNames);


        if(!windowCreated) {
            cv::namedWindow(windowName, 0);
            if(isFullScreen)
                cv::setWindowProperty(windowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        }
        cv::imshow(windowName, image);
        auto finish = std::chrono::steady_clock::now();

        auto key = cv::waitKey(isVideo) & 0xFF;
        if(key == 'q' && isVideo)
            break;

        if(isVideo)
            finish = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();

        if(isVideo) {
            avgduration = (avgalpha * duration) + (1.0 - avgalpha) * avgduration;
            std::cout << 1.0 / (avgduration + 0.0001) << " fps" << std::endl;
        } else {
            std::cout << "Elapsed: " << duration << " seconds" << std::endl;
        }
    } while(isVideo);

    return 0;
}

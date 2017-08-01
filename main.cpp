#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include "mtcnn/mtcnn.h"
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>


using namespace cv;
using namespace std;

static dlib::rectangle openCVRectToDlib(cv::Rect r) {
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

int main(int argc, char **argv)
{

    dlib::shape_predictor pose_model;
    dlib::deserialize("lib/shape_predictor_68_face_landmarks.dat") >> pose_model;
    std::vector<dlib::rectangle> dlibFaces;
    std::vector<dlib::full_object_detection> shapes;

    // "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"
    std::vector<Ptr<Tracker>> trackers;

    // Read video
    VideoCapture video(0);

    // Exit if video is not opened
    if(!video.isOpened()) return 1;

    // Read first frame
    Mat frame;
    bool ok = video.read(frame);

    mtcnn faceDetector(frame.rows, frame.cols);

    faceDetector.findFace(frame);
    // Rect2d bbox = faceDetector.getFaceBoxes()[0];
    std::vector<cv::Rect2d> bbox = faceDetector.getFaceBoxes();
    int face_count = 0;
    for (const auto& b: bbox) {
        rectangle(frame, b, Scalar( 255, 0, 0 ), 2, 1 );
        trackers.push_back(Tracker::create("KCF"));
        trackers[face_count]->init(frame, b);
        face_count++;
    }

    // tracker->init(frame, bbox);

    int count = 0;
    for(;;) {

        video >> frame;

        if (faceDetector.getFaceBoxes().empty() || count % 5 == 0) {
            trackers.clear();
            faceDetector.findFace(frame);
            bbox = faceDetector.getFaceBoxes();
            int face_count = 0;
            for (const auto& b: bbox) {
                cv::rectangle(frame, b, Scalar(0,0,255), 2,8,0);
                trackers.push_back(Tracker::create("KCF"));
                trackers[face_count]->init(frame, b);
                face_count++;
            }

        }
        else {
            int face_count = 0;
            for (auto& b: bbox) {
                bool ok = trackers[face_count]->update(frame, b);
                rectangle(frame, b, Scalar( 255, 0, 0 ), 2, 1 );
                face_count++;
            }
        }

        dlib::cv_image<dlib::bgr_pixel>  cimg(frame);
        for (const auto& b : bbox) {
            dlibFaces.push_back(openCVRectToDlib(b));
        }
        for (const auto& b : dlibFaces) {
            shapes.push_back(pose_model(cimg, b));
        }

        for (const auto& shape : shapes) {
            for (int i = 0; i < shape.num_parts(); i++) {
                cv::circle(frame, cv::Point(shape.part(i).x(), shape.part(i).y()), 3, cv::Scalar(50,0,0), -1);
            }
        }

        imshow("result", frame);
        shapes.clear();
        dlibFaces.clear();
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        count++;
    }

    waitKey(0);
    frame.release();
    return 0;
}
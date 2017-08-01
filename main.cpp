#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "mtcnn/mtcnn.h"


using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
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

        if (faceDetector.getFaceBoxes().empty() || count % 10 == 0) {
            faceDetector.findFace(frame);
            bbox = faceDetector.getFaceBoxes();
            for (const auto& b: bbox) {
                cv::rectangle(frame, b, Scalar(0,0,255), 2,8,0);
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

        imshow("result", frame);
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
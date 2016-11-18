#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/*

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
				   CascadeClassifier& nested_cascade,
				   double scale, bool tryflip);

int main(int argc, char *args[]) {
	string input_name;
	double scale = 1;
	bool tryflip = false;


	VideoCapture webcam;
	Mat frame, image;
	string cascade_name = "data/haarcascades/haarcascade_frontalface_default.xml";
	string nested_cascade_name = "data/haarcascades/haarcascade_eye.xml";

	CascadeClassifier cascade, nested_cascade;

    if (!nested_cascade.load(nested_cascade_name))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;

    if (!cascade.load(cascade_name)) {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        // help();
        return -1;
    }
    if (input_name.empty() || (isdigit(input_name[0]) && input_name.size() == 1)) {
        int c = input_name.empty() ? 0 : input_name[0] - '0';
        if (!webcam.open(c))
            cout << "Capture from camera #" << c << " didn't work" << endl;
    } else if (input_name.size()) {
        image = imread(input_name, 1);
        if (image.empty()) {
            if (!webcam.open(input_name))
                cout << "Could not read " << input_name << endl;
        }
    } else {
        image = imread("../data/lena.jpg", 1);
        if (image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
    }

    if (webcam.isOpened()) {
        cout << "Video capturing has been started ..." << endl;
        for (;;) {
            webcam >> frame;
            if (frame.empty())
                break;

            Mat frame1 = frame.clone();
            detectAndDraw(frame1, cascade, nested_cascade, scale, tryflip);

            int c = waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    } else {
        cout << "Detecting face(s) in " << input_name << endl;
        if (!image.empty()) {
            detectAndDraw(image, cascade, nested_cascade, scale, tryflip);
            waitKey(0);
        } else if (!input_name.empty()) {
            FILE* f = fopen(input_name.c_str(), "rt");
            if (f) {
                char buf[1000 + 1];
                while (fgets(buf, 1000, f)) {
                    int len = (int)strlen(buf), c;
                    while (len > 0 && isspace(buf[len - 1]))
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread(buf, 1);
                    if (!image.empty()) {
                        detectAndDraw(image, cascade, nested_cascade, scale, tryflip);
                        c = waitKey(0);
                        if (c == 27 || c == 'q' || c == 'Q')
                            break;
                    } else {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
                   CascadeClassifier& nested_cascade,
                   double scale, bool tryflip) {
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] = {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scale;
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);
    t = (double)cvGetTickCount();
    cascade.detectMultiScale(smallImg, faces,
                             1.1, 2, 0
                             | CASCADE_FIND_BIGGEST_OBJECT
                             | CASCADE_DO_ROUGH_SEARCH
                             // | CASCADE_SCALE_IMAGE
							 ,
                             Size(30, 30));
	
    if (tryflip) {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale(smallImg, faces2,
                                 1.1, 2, 0
                                 | CASCADE_FIND_BIGGEST_OBJECT
                                 | CASCADE_DO_ROUGH_SEARCH
                                 // | CASCADE_SCALE_IMAGE
								 ,
                                 Size(30, 30));
        for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++) {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }

    t = (double)cvGetTickCount() - t;
    printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));
    for (size_t i = 0; i < faces.size(); i++) {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i % 8];
        int radius;
		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3) {
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		} else
			rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
						cvPoint(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
						color, 3, 8, 0);
		continue;
		if (nested_cascade.empty())
			continue;
		smallImgROI = smallImg(r);
		nested_cascade.detectMultiScale(smallImgROI, nestedObjects,
										1.1, 2, 0
										|CASCADE_FIND_BIGGEST_OBJECT
										// |CASCADE_DO_ROUGH_SEARCH
										// |CASCADE_DO_CANNY_PRUNING
										// | CASCADE_SCALE_IMAGE
										, Size(30, 30));
		for (size_t j = 0; j < nestedObjects.size(); j++) {
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
			center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
			radius = cvRound((nr.width + nr.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
    }
    imshow("result", img);
}

*/
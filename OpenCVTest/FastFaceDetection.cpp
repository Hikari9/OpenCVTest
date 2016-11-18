#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
typedef vector<CascadeClassifier> CascadeList;
void face_detection(Mat& img, CascadeList& features, double scale);
void log_statistics(double stat) {
	return;
	static ofstream stream("stat.out");
	stream.precision(6);
	stream << fixed << stat << endl;
}
int main() {
	int camera_id = 0;
	VideoCapture webcam;
	Mat frame, image;
	string folder = "data/haarcascades/";
	vector<string> features = {
		"haarcascade_eye.xml",
	};
	CascadeList cascades(features.size());
	for (int i = 0; i < features.size(); ++i)
		if (!cascades[i].load(folder + features[i]))
			return cerr << "Warning: could not open cascade " << folder << features[i] << endl, -1;

	if (!webcam.open(camera_id))
		return cerr << "Cannot capture from camera #" << camera_id << endl, -1;

	if (webcam.isOpened()) {
		cout << "Video capturing has been started ..." << endl;
		while (true) {
			webcam >> frame;
			if (frame.empty())
				break;
			face_detection(frame, cascades, 1.0);
			int c = waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	return 0;
}
namespace filter {
	void grayscale(Mat& img) {
		// converts image to grayscale
		cvtColor(img, img, COLOR_BGR2GRAY);
	}
	void scale(Mat& img, double factor) {
		// resize and scale image by a factor
		double fx = 1.0 / factor;
		Mat small;
		resize(img, small, Size(), fx, fx, INTER_LINEAR);
		img = small;
	}
	void equalize(Mat& img) {
		// equalize intensity histograms
		equalizeHist(img, img);
	}
}
void face_detection(Mat& img, CascadeList& cascades, double scale) {

	Mat original = img.clone();

	int64 filter_start = cvGetTickCount();
	filter::scale(img, scale);
	filter::grayscale(img);
	filter::equalize(img);
	double filter_time = (cvGetTickCount() - filter_start) / (cvGetTickFrequency() * 1000.0);

	int64 cascade_start = cvGetTickCount();
	vector<Rect> faces;
	int cascade_flags = 0
		// | CASCADE_DO_ROUGH_SEARCH
		| CASCADE_DO_CANNY_PRUNING
		// | CASCADE_FIND_BIGGEST_OBJECT
		// | CASCADE_SCALE_IMAGE
		;
	for (auto& cascade : cascades) {
		vector<Rect> cascadeFaces;
		cascade.detectMultiScale(img, cascadeFaces, 1.1, 2, cascade_flags, Size(30, 30));
		faces.insert(faces.end(), cascadeFaces.begin(), cascadeFaces.end());
	}

	double cascade_time = (cvGetTickCount() - cascade_start) / (cvGetTickFrequency() * 1000.0);
	printf("face_detection time = %.4lfms\n", cascade_time + filter_time);
	log_statistics(cascade_time + filter_time);
	for (auto& face : faces) {
		Scalar color(255, 0, 0);
		Mat region_of_interest;
		rectangle(img, face, color, 3);
		// Point center(cvRound((face.x + face.width / 2.0) * scale), cvRound((face.y + face.height / 2.0) * scale));
		// int radius = 3;
		// circle(img, center, radius, color, 3);
	}
	imshow("result", img);

}
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame, CascadeClassifier faceCascade,
                      CascadeClassifier smileCascade);

string faceCascade;
string smileCascade;

int main(int argc, const char** argv)
{
    VideoCapture videoCapture;
    Mat frame, image;

    CascadeClassifier faceCascadeCl, smileCascadeCl;
    CommandLineParser parser(argc, argv,
            "{frontalface|data/haarcascades/haarcascade_frontalface_alt.xml|}"
            "{smile|data/haarcascades/haarcascade_smile.xml|}");

    // Load the cascades
    faceCascade = samples::findFile(parser.get<string>("frontalface"));
    smileCascade = samples::findFile(parser.get<string>("smile"));

    if(!faceCascadeCl.load(faceCascade))
    {
        cerr << "ERROR: Could not load face cascade" << endl;
        return -1;
    }
    if(!smileCascadeCl.load( smileCascade))
    {
        cerr << "ERROR: Could not load smile cascade" << endl;
        return -1;
    }

    videoCapture.open(0);
    if( videoCapture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;

        for(;;)
        {
            videoCapture >> frame;
            if( frame.empty() )
                break;

            Mat frameClone = frame.clone();
            detectAndDisplay( frameClone, faceCascadeCl, smileCascadeCl);

            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    else
    {
        cerr << "ERROR: Could not initiate capture" << endl;
        return -1;
    }
    videoCapture.release();
    destroyAllWindows();
    return 0;
}

// A function that detects face and the smile
void detectAndDisplay(Mat frame, CascadeClassifier faceCascade,
                      CascadeClassifier smileCascade)
{
    vector<Rect> faces;
    Mat gray;

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    // These parameters are decided after some tests and there are the best fits
    faceCascade.detectMultiScale( gray, faces,
                             1.3, 5, 0
                             //|CASCADE_FIND_BIGGEST_OBJECT
                             //|CASCADE_DO_ROUGH_SEARCH
                             |CASCADE_SCALE_IMAGE,
                             Size(30, 30) );

    for (size_t i = 0; i<faces.size(); i++)
    {
        Point upperLeftCorner(faces[i].x, faces[i].y);
        Point lowerRightCorner(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

        // Draw rectangle around the face
        rectangle(frame, upperLeftCorner, lowerRightCorner, Scalar(255, 0, 0), 2);

        // Get only the face from the detection
        // To improve the performance, we are bounding the search area.
        // We will only search inside the face
        Mat frameROI = frame(faces[i]);
        Mat grayROI = gray(faces[i]);
        vector<Rect> smile;

        // Lower number of neighbours means it will detect everthing that is similiar to a a smile
        smileCascade.detectMultiScale(grayROI, smile, 1.8, 20);
        for (size_t j = 0; j < smile.size(); j++)
        {
            // Add text if there is a smile
            putText(frameROI, "Smiling", Point(10, (smile[j].width/2)), FONT_HERSHEY_SIMPLEX, 2,
                                               Scalar(255, 255, 255), 2, LINE_AA);
        }
    }
    imshow("Video", frame);
}

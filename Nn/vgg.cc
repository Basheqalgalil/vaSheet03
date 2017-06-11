#include "vgg.hh"


 static const Core::ParameterString paramVideosDirectory_("videos-directory","","vgg");

vgg::vgg():
    VideosDirectory_(Core::Configuration::config(paramVideosDirectory_))
{
}
void prepareVideoForTemporalStream(){

}
void createTemporalFramesFromVideo(const std::string& filename){
   Video video ;
   readVideo(filename,video);



}
void vgg::readVideo(const std::string& filename, Video& result) {
    // open video file
    cv::VideoCapture capture(filename);
    if(!capture.isOpened())
        Core::Error::msg() << "Unable to open Video: " << filename << Core::Error::abort;
    cv::Mat frame, tmp;
    result.clear();
    // read all frames
    u32 nFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    while ((nFrames > 0) && (capture.read(frame))) {
        if (frame.channels() == 3)
            cv::cvtColor(frame, tmp, CV_BGR2GRAY);
        else
            tmp = frame;
        result.push_back(cv::Mat());
        tmp.convertTo(result.back(), CV_32FC1, 1.0/255.0);
        nFrames--;
    }
    capture.release();
}

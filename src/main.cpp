#include <vector>
#include <ostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "net.h"
#include "cpu.h"
#include "gfpgan.h"
#include "face.h"
#include "realesrgan.h"

using namespace std;
//------------------------------------------------------------------------------------
static void to_ocv( const ncnn::Mat& result, cv::Mat& out)
{
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);
    for (int i = 0; i < result.h; i++)
    {
        for (int j = 0; j < result.w; j++)
        {
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = (result.channel(0)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = (result.channel(1)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = (result.channel(2)[i * result.w + j] + 1) / 2;
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);

    cv_result_8U.copyTo(out);

}
//------------------------------------------------------------------------------------
static void paste_faces_to_input_image(const cv::Mat& restored_face,cv::Mat& trans_matrix_inv,cv::Mat& bg_upsample)
{
    trans_matrix_inv.at<float>(0, 2) += 1.0;
    trans_matrix_inv.at<float>(1, 2) += 1.0;

    cv::Mat inv_restored;
    cv::warpAffine(restored_face, inv_restored, trans_matrix_inv, bg_upsample.size(), 1, 0);
    cv::Mat mask = cv::Mat::ones(cv::Size(512, 512), CV_8UC1) * 255;
    cv::Mat inv_mask;
    cv::warpAffine(mask, inv_mask, trans_matrix_inv, bg_upsample.size(), 1, 0);
    cv::Mat inv_mask_erosion;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
    cv::erode(inv_mask, inv_mask_erosion, kernel);
    cv::Mat pasted_face;
    cv::bitwise_and(inv_restored, inv_restored, pasted_face, inv_mask_erosion);

    int total_face_area = cv::countNonZero(inv_mask_erosion);
    int w_edge = int(std::sqrt(total_face_area) / 20);
    int erosion_radius = w_edge * 2;
    cv::Mat inv_mask_center;
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erosion_radius, erosion_radius));
    cv::erode(inv_mask_erosion, inv_mask_center, kernel);

    int blur_size = w_edge * 2;
    cv::Mat inv_soft_mask;
    cv::GaussianBlur(inv_mask_center, inv_soft_mask, cv::Size(blur_size + 1, blur_size + 1), 0, 0, 4);

    for (int h = 0; h < bg_upsample.rows; h++)
    {
        for (int w = 0; w < bg_upsample.cols; w++)
        {
            float alpha = inv_soft_mask.at<uchar>(h, w) / 255.0;
            bg_upsample.at<cv::Vec3b>(h, w)[0] = pasted_face.at<cv::Vec3b>(h, w)[0] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[0];
            bg_upsample.at<cv::Vec3b>(h, w)[1] = pasted_face.at<cv::Vec3b>(h, w)[1] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[1];
            bg_upsample.at<cv::Vec3b>(h, w)[2] = pasted_face.at<cv::Vec3b>(h, w)[2] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[2];
        }
    }
}
//------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    float Ratio=0.0;
    bool WholePicture=false;
    cv::Mat bg_upsample;
    cv::Mat restored_face;
    vector<Object>  objects;
    vector<cv::Mat> trans_img;
    vector<cv::Mat> trans_matrix_inv;
    // ncnn models
    Face face_detector;
    RealESRGAN real_esrgan;
    GFPGAN gfpgan;
    //some timning
    std::chrono::steady_clock::time_point Tbegin, Tend;

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    cout << "Loading networks ...." << endl;

    face_detector.load("./models/yolov5-blazeface.param", "./models/yolov5-blazeface.bin");
    gfpgan.load("./models/encoder.param", "./models/encoder.bin", "./models/style.bin");
    real_esrgan.load("./models/real_esrgan.param", "./models/real_esrgan.bin");

    Tbegin = chrono::steady_clock::now();

    //get the faces
    face_detector.detect(img, objects, 0.35);
    if(objects.size() > 1) {
        WholePicture=true;
    }
    else{
        if(objects.size() == 1){
            Ratio =(objects[0].rect.width * objects[0].rect.height);
            Ratio/=(img.cols * img.rows);
            if(Ratio < 0.1) WholePicture=true;
        }
    }
    //restore whole picture if there are more faces, or if the only face is in the background (Ratio < 0.1)
    if(WholePicture){
        cout << "Start restoring the whole image..." << endl;

        real_esrgan.tile_process(img, bg_upsample);

        face_detector.align_warp_face(img, objects, trans_matrix_inv, trans_img);
        for(size_t i = 0; i < objects.size(); i++) {
            ncnn::Mat gfpgan_result;
            gfpgan.process(trans_img[i], gfpgan_result);

            to_ocv(gfpgan_result, restored_face);

            paste_faces_to_input_image(restored_face, trans_matrix_inv[i], bg_upsample);
        }

        //calculate inference time
        Tend = chrono::steady_clock::now();
        int f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        cout << "Inference time : " << f/1000.0 <<  " Sec" << endl;

        //show and save result
        cv::imwrite("result.png",bg_upsample);
        cv::imshow("Restored", bg_upsample);
        cv::waitKey();
    }
    else{
        cout << "Start restoring the face only..." << endl;

        ncnn::Mat gfpgan_result;
        gfpgan.process(img, gfpgan_result);

        to_ocv(gfpgan_result, restored_face);

        //calculate inference time
        Tend = chrono::steady_clock::now();
        int f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        cout << "Inference time : " << f/1000.0 <<  " Sec" << endl;

        //show and save result
        cv::imwrite("result.png",restored_face);
        cv::imshow("Restored", restored_face);
        cv::waitKey();
    }
    return 0;
}
//------------------------------------------------------------------------------------


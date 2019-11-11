//
//  main.cpp
//  siftassign
//
//  Created by Winter Cyan on 2019/11/9.
//  Copyright © 2019 Winter Cyan. All rights reserved.
//

#include <iostream>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxcore.h>
#include <math.h>
#include <string>

#include "sift.hpp"
#include "sift.cpp"
#include "imgfeatures.hpp"
#include "imgfeatures.cpp"
#include "kdtree.hpp"
#include "kdtree.cpp"
#include "utils.hpp"
#include "utils.cpp"
#include "xform.hpp"
#include "xform.cpp"
#include "minpq.hpp"
#include "minpq.cpp"

using namespace cv;
using namespace std;

#define KDTREE_BBF_MAX_NN_CHKS 200
#define CAMERA_F 1750
#define NN_SQ_DIST_RATIO_THR 0.49

void disp_img(IplImage* img, char* title);
void CalcCorners(const Mat& H, const Mat& src);
IplImage* stack_imgs_row( IplImage* img1, IplImage* img2 );
IplImage* stitch_two(IplImage* img1, IplImage* img2);
IplImage* proj_to_cylind(IplImage* src, double f);
IplImage* panorama(String dir, int n);
Mat elimita_H(Mat mat_H);
void output_H(Mat& H);
typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;
 
four_corners_t corners;

int main() {
//    IplImage* img1 = cvLoadImage("/Users/wintercyan/Pictures/pano/pairs/lf1.jpeg", 1);
//    IplImage* img2 = cvLoadImage("/Users/wintercyan/Pictures/pano/pairs/lf2.jpeg", 1);
//    IplImage* stitched = stitch_two(img1, img2);
//    disp_img(stitched, "final");
    
//    IplImage* img1_src, *img2_src, *img3_src, *img1, *img2, *img3, *img_s1, *img_s2;
//    img1_src = cvLoadImage( "/Users/wintercyan/Pictures/pano/desk/_1.jpg", 1 );
//    img2_src = cvLoadImage( "/Users/wintercyan/Pictures/pano/desk/0.jpg", 1 );
//    img3_src = cvLoadImage( "/Users/wintercyan/Pictures/pano/desk/1.jpg", 1 );
//    img1 = proj_to_cylind(img1_src, CAMERA_F);
//    img2 = proj_to_cylind(img2_src, CAMERA_F);
//    img3 = proj_to_cylind(img3_src, CAMERA_F);
//    img_s1 = stitch_two(img1, img2);
//    img_s2 = stitch_two(img_s1, img3);
////    img_s1 = stitch_two(img1_src, img2_src);
////    img_s2 = stitch_two(img_s1, img3_src);
//    disp_img(img_s1, "final1");
//    disp_img(img_s2, "final2");
//
    IplImage* pano = panorama("/Users/wintercyan/Pictures/pano/groups/g3", 10);
    disp_img(pano, "panorama");
    
    return 0;
}

IplImage* stitch_two(IplImage* img1, IplImage* img2) {
//    IplImage * stacked;
    struct feature* feat1, * feat2, * feat;
    struct feature** nbrs;
    struct kd_node* kd_root;
    int n1, n2;
    
//    stacked = stack_imgs_row( img1, img2 );

    cout<<"finding features in the first image...: ";
    n1 = sift_features( img1, &feat1 );
    cout<<n1<<" features found"<<endl;
    cout<<"finding features in the second image...: ";
    n2 = sift_features( img2, &feat2 );
    cout<<n2<<" features found"<<endl<<endl;
    cout<<"building kd-tree and acting RANSAC..."<<endl;

    kd_root = kdtree_build( feat1, n1 );
    Point pt1,pt2;
    double d0,d1;
    int matchNum = 0;
    //kd-tree match
    for(int i = 0; i < n2; i++ ) {
        feat = feat2+i;
        int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
        if(k == 2) {
            d0 = descr_dist_sq( feat, nbrs[0] );
            d1 = descr_dist_sq( feat, nbrs[1] );
            if( d0 < d1 * NN_SQ_DIST_RATIO_THR ) {
//                pt2 = Point( cvRound( feat->x ), cvRound( feat->y ) );
//                pt1 = Point( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
//                pt2.x += img1->width;
//                cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
                matchNum++;
                feat2[i].fwd_match = nbrs[0];
            }
        }
        free(nbrs);
    }
    
    CvMat* H;
    struct feature** inliers;
    int n_inliers;
    H = ransac_xform(feat2,n2,FEATURE_FWD_MATCH,lsq_homog,4,0.01,homog_xfer_err,3.0,&inliers,&n_inliers);
    if(H) {
        cout<<"matched features after RANSAC："<<n_inliers<<endl<<endl;
        
        // left-right order
        int invertNum = 0;
        for(int i=0; i<n_inliers; i++) {
            feat = inliers[i];
            pt2 = Point(cvRound(feat->x), cvRound(feat->y));
            pt1 = Point(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));
            if(pt2.x > pt1.x) invertNum++;
            pt2.x += img1->width;
//            cvLine(stacked,pt1,pt2,CV_RGB(255,0,255),1,8,0);
        }
        if(invertNum > n_inliers * 0.8) {
            CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);
            if( cvInvert(H,H_IVT) ) {
                cvReleaseMat(&H);
                H = cvCloneMat(H_IVT);
                cvReleaseMat(&H_IVT);
                IplImage * temp = img2;
                img2 = img1;
                img1 = temp;
            }
            else {
                cvReleaseMat(&H_IVT);
                cout<<"non-invertible"<<endl;
                return NULL;
            }
        }
        
        // stitch and blend
        Mat mat_H = cvarrToMat(H);
        cout<<"the homography matrix H:"<<endl;
        output_H(mat_H);
        Mat mat_Img = cvarrToMat(img2);
        CalcCorners(mat_H, mat_Img);
        IplImage* xformed, *xformed_add, * xformed_blend;
        xformed = cvCreateImage(cvSize(MIN(corners.right_top.x,corners.right_bottom.x),MIN(img1->height,img2->height)),IPL_DEPTH_8U,3);
        cvWarpPerspective(img2,xformed,H,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll(0));
        xformed_add = cvCloneImage(xformed);
        cvSetImageROI(xformed_add,cvRect(0,0,img1->width,img1->height));
        cvAddWeighted(img1,1,xformed_add,0,0,xformed_add);
        cvResetImageROI(xformed_add);
        xformed_blend = cvCloneImage(xformed);
        cvSetImageROI(img1,cvRect(0,0,MIN(corners.left_top.x,corners.left_bottom.x),xformed_blend->height));
        cvSetImageROI(xformed,cvRect(0,0,MIN(corners.left_top.x,corners.left_bottom.x),xformed_blend->height));
        cvSetImageROI(xformed_blend,cvRect(0,0,MIN(corners.left_top.x,corners.left_bottom.x),xformed_blend->height));
        cvAddWeighted(img1,1,xformed,0,0,xformed_blend);
        cvResetImageROI(img1);
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_blend);
        int blend_left = MIN(corners.left_top.x,corners.left_bottom.x) ;
        double blend_width = img1->width - blend_left;
        double blend_alpha = 1;
        for(int i=0; i<xformed_blend->height; i++) {
            const uchar * row_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));
            const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));
            uchar * pixel_blend = ((uchar *)(xformed_blend->imageData + xformed_blend->widthStep * i));
            for(int j=blend_left; j<img1->width; j++) {
                if(pixel_xformed[j*3] < 50 && pixel_xformed[j*3+1] < 50 && pixel_xformed[j*3+2] < 50 ) blend_alpha = 1;
                else blend_alpha = (blend_width-(j-blend_left)) / blend_width ;
                pixel_blend[j*3] = row_img1[j*3] * blend_alpha + pixel_xformed[j*3] * (1-blend_alpha);
                pixel_blend[j*3+1] = row_img1[j*3+1] * blend_alpha + pixel_xformed[j*3+1] * (1-blend_alpha);
                pixel_blend[j*3+2] = row_img1[j*3+2] * blend_alpha + pixel_xformed[j*3+2] * (1-blend_alpha);
            }
        }
//        cvReleaseImage( &stacked );
        cvReleaseImage( &img1 );
        cvReleaseImage( &img2 );
        kdtree_release( kd_root );
        free( feat1 );
        free( feat2 );
        return xformed_blend;
    }
    else {
        cout<<"no matched area"<<endl;
        return NULL;
    }
}

void disp_img(IplImage* img, char* title) {
    cvNamedWindow(title, WINDOW_FREERATIO);
    cvShowImage(title, img);
    cvWaitKey(0);
}
 
void CalcCorners(const Mat& H, const Mat& src) {
    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
 
    V1 = H * V2;
    //左上角(0,0,1)
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];
 
    //左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];
 
    //右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];
 
    //右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];
    cout<<"four corners after trans:"<<endl;
    cout<<"("<<corners.left_top.x<<", "<<corners.left_top.y<<")  "<<"("<<corners.right_top.x<<", "<<corners.right_top.y<<")"<<endl;
    cout<<"("<<corners.left_bottom.x<<", "<<corners.left_bottom.y<<")  "<<"("<<corners.right_bottom.x<<", "<<corners.right_bottom.y<<")"<<endl;
}

//IplImage* stack_imgs_row( IplImage* img1, IplImage* img2 ) {
//  IplImage* stacked = cvCreateImage( cvSize(img1->width+img2->width, MAX(img1->height, img2->height)),IPL_DEPTH_8U, 3 );
//
//  cvZero( stacked );
//  cvSetImageROI( stacked, cvRect( 0, 0, img1->width, img1->height ) );
//  cvAdd( img1, stacked, stacked, NULL );
//  cvSetImageROI( stacked, cvRect(img1->width, 0, img2->width+img1->width, img2->height) );
//  cvAdd( img2, stacked, stacked, NULL );
//  cvResetImageROI( stacked );
//
//  return stacked;
//}

IplImage* proj_to_cylind(IplImage* src, double f) {
    int width = src->width;
    int height = src->height;
    IplImage* img_cylind = cvCreateImage(cvSize(width, height), src->depth, src->nChannels);
    int x0, y0;
    for (int x0_ = 0; x0_ < width; x0_ ++) {
        for (int y0_ = 0; y0_ < height; y0_ ++) {
            double x = f*tan((x0_-width/2.0)/f);
            double x0f = x + width/2.0;
            double y0f = height/2.0 - ((height/2.0-y0_)*sqrt(x*x+f*f))/f;
            x0 = cvRound(x0f);
            y0 = cvRound(y0f);
            const uchar * pixel_src = ((uchar *)(src->imageData + src->widthStep * y0));
            uchar * pixel_cylind = ((uchar *)(img_cylind->imageData + img_cylind->widthStep * y0_));
            if (x0 <= 0 || y0 <= 0 || x0 >= width || y0 >= height) {
                pixel_cylind[x0_*3] = 0;
                pixel_cylind[x0_*3 + 1] = 0;
                pixel_cylind[x0_*3 + 2] = 0;
            } else {
                pixel_cylind[x0_*3] = pixel_src[x0*3];
                pixel_cylind[x0_*3 + 1] = pixel_src[x0*3 + 1];
                pixel_cylind[x0_*3 + 2] = pixel_src[x0*3 + 2];
            }
        }
    }
    return img_cylind;
}

IplImage* panorama(String dir, int n) {
    IplImage* src_img, *new_img, *stitched_img;
//    cvZero(stitched_img);
    for (int i = 1; i < n; i ++) {
        String name = to_string(i+1) + ".jpg";
        src_img = stitched_img;
        if (i == 1) src_img = proj_to_cylind(cvLoadImage((dir+"/"+to_string(i) + ".jpg").c_str()), CAMERA_F);
        new_img = proj_to_cylind(cvLoadImage((dir+"/"+name).c_str()), CAMERA_F);
        cout<<endl<<"--------ITERATION "<<i<<" OF STITCHING--------"<<endl;
        stitched_img = stitch_two(src_img, new_img);
    }
    cout<<"--------STITCH OVER--------"<<endl<<endl<<endl;
    return stitched_img;
}

//Mat elimita_H(Mat mat_H) {
//    double r1, r2, r3, r4, tx, ty, r1_, r2_, r3_, r4_, tx_, ty_;
//    r1 = mat_H.at<double>(0, 0);
//    r2 = mat_H.at<double>(0, 1);
//    r3 = mat_H.at<double>(1, 0);
//    r4 = mat_H.at<double>(1, 1);
//    tx = mat_H.at<double>(0, 2);
//    ty = mat_H.at<double>(1, 2);
//
//    r1_ = r1;
//    r4_ = r4-(r3*r2)/r1;
//    tx_ = tx-((ty-r3*tx/r1)*tx)/(r4-r3*r2/r1);
//    ty_ = ty-r3*tx/r1;
//
//    mat_H.at<double>(0, 0) = 1;
//    mat_H.at<double>(0, 1) = 0;
//    mat_H.at<double>(1, 0) = 0;
//    mat_H.at<double>(1, 1) = 1;
//    mat_H.at<double>(0, 2) = abs(tx_/r1_);
//    mat_H.at<double>(1, 2) = abs(ty_/r4_);
//
//    return mat_H;
//}

void output_H(Mat& H) {
    cout<<"["<<H.at<double>(0,0)<<", "<<H.at<double>(0,1)<<", "<<H.at<double>(0,2)<<"]"<<endl;
    cout<<"["<<H.at<double>(1,0)<<", "<<H.at<double>(1,1)<<", "<<H.at<double>(1,2)<<"]"<<endl;
    cout<<"["<<H.at<double>(2,0)<<", "<<H.at<double>(2,1)<<", "<<H.at<double>(2,2)<<"]"<<endl<<endl;
}

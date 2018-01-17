//
//  Created by Melvin James on 2/7/17.
//  Copyright © 2017 Melvin James. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
Mat image,gaussian_noise,N1,H1,SnP_noise,N2,H2,B1,temp;

void DenoiseSnP()
{
    const int kernalWidth=3;
    const int kernalHeight=3;
    int size =kernalWidth*kernalHeight;
    float kernalArray[kernalWidth][kernalHeight] = {{0}};

    int rows=N2.rows;
    int cols=N2.cols;
    
    Mat DenoisedImage(N2.size(),N2.type());

    
    
    for(int row=0+1; row<rows-1; row++){
        
        for(int col=0+1;col<cols-1;col++){
            
            float valuer[9] = {0},valueg[9] = {0},valueb[9] = {0};
            int i =0;
            for(int kRow=0;kRow<kernalHeight;kRow++){
                for(int kCol=0;kCol<kernalWidth;kCol++){
                    //               multiply pixel value with corresponding gaussian kernal value
                    float pixelr=N2.at<Vec3b>(kRow+row-1,kCol+col-1)[2]+kernalArray[kRow][kCol];
                    valuer[i]=pixelr;
                    float pixelg=N2.at<Vec3b>(kRow+row-1,kCol+col-1)[1]+kernalArray[kRow][kCol];
                    valueg[i]=pixelg;
                    float pixelb=N2.at<Vec3b>(kRow+row-1,kCol+col-1)[0]+kernalArray[kRow][kCol];
                    valueb[i]=pixelb;
                    i++;
                }
            }
            sort(valuer,valuer+size);
            sort(valueg,valueg+size);
            sort(valueb,valueb+size);
            //assign new values to central point
            DenoisedImage.at<Vec3b>(row,col)[2]=(valuer[5]);
            DenoisedImage.at<Vec3b>(row,col)[1]=(valueg[5]);
            DenoisedImage.at<Vec3b>(row,col)[0]=(valueb[5]);
        }
    }
    H2 = DenoisedImage;
    namedWindow("filtered S and P H2", 1);
    
    imshow("filtered S and P H2", H2);
   
    
}
void DenoiseG(Mat N1)
{
    
const double PI = 3.14 ;
double sigma=  1;
const int kernalWidth=5;
const int kernalHeight=5;

float kernalArray[kernalWidth][kernalHeight];

double total=0;

//calculate each relavant value to neighour pixels and store it in 2d array
for(int row=0;row<kernalWidth;row++){
    
    for(int col=0;col<kernalHeight;col++){
        
        float value=(1/(2*PI*pow(sigma,2)))*exp(-(pow(row-kernalWidth/2,2)+pow(col-kernalHeight/2,2))/(2*pow(sigma,2)));
        
        kernalArray[row][col]=value;
        
        total+=value;
    }
}
for(int row=0;row<kernalWidth;row++){
    for(int col=0;col<kernalHeight;col++){
        
        kernalArray[row][col]=kernalArray[row][col]/total;
        
    }
}

Mat DenoisedImage(N1.size(),N1.type());

int rows=N1.rows;
int cols=N1.cols;


int verticleImageBound=(kernalHeight-1)/2;
int horizontalImageBound=(kernalWidth-1)/2;


for(int row=0+verticleImageBound; row<rows-verticleImageBound; row++){
    
    for(int col=0+horizontalImageBound;col<cols-horizontalImageBound;col++){
        
        float valuer =0.0,valueg=0.0, valueb=0.0;
        
        for(int kRow=0;kRow<kernalHeight;kRow++){
            for(int kCol=0;kCol<kernalWidth;kCol++){
 //               multiply pixel value with corresponding gaussian kernal value
                float pixelr=N1.at<Vec3b>(kRow+row-verticleImageBound,kCol+col-horizontalImageBound)[2]*kernalArray[kRow][kCol];
                valuer+=pixelr;
                float pixelg=N1.at<Vec3b>(kRow+row-verticleImageBound,kCol+col-horizontalImageBound)[1]*kernalArray[kRow][kCol];
                valueg+=pixelg;
                float pixelb=N1.at<Vec3b>(kRow+row-verticleImageBound,kCol+col-horizontalImageBound)[0]*kernalArray[kRow][kCol];
                valueb+=pixelb;
            }
        }
        //assign new values to central point
        DenoisedImage.at<Vec3b>(row,col)[2]=(valuer);
        DenoisedImage.at<Vec3b>(row,col)[1]=(valueg);
        DenoisedImage.at<Vec3b>(row,col)[0]=(valueb);
    }
}
    H1 = DenoisedImage;
    namedWindow("Gauss filtered H1", 1);

    imshow("Gauss filtered H1", H1);
}
void GaussN()                                                   //Add Gaussian noise
{
    gaussian_noise = temp.clone();
    N1 = temp;
    randn(gaussian_noise,0,15);
    N1+=gaussian_noise;
    namedWindow("Image with Gaussian noise N1", 1);
    
    imshow("Image with Gaussian noise N1", N1);
    DenoiseG(N1);
    
}
void SaltnPepper()
{
    SnP_noise = Mat::zeros(image.rows, image.cols,CV_8U);
    N2 = image.clone();
    
//    randu(SnP_noise,0,255);
//    
//    Mat black = SnP_noise < 30;
//    Mat white = SnP_noise > 245;
//    
//    
//    N2.setTo(255,white);
//    N2.setTo(0,black);
    
    int specks = ceil(0.01*image.rows*image.cols*0.5);
    for(int i = 0; i< specks; i++)
    {
        int y = rand()%(image.rows - 1);
        int x = rand()%(image.cols - 1);
        N2.at<Vec3b>(y,x)[0] = 255;
        N2.at<Vec3b>(y,x)[1] = 255;
        N2.at<Vec3b>(y,x)[2] = 255;
    }
    for(int i = 0; i< specks; i++)
    {
        int y = rand()%(image.rows - 1);
        int x = rand()%(image.cols - 1);
        N2.at<Vec3b>(y,x)[0] = 0;
        N2.at<Vec3b>(y,x)[1] = 0;
        N2.at<Vec3b>(y,x)[2] = 0;
    }
    
    namedWindow("Image with Salt and Pepper noise N2", 1);
    
    imshow("Image with Salt and Pepper noise N2", N2);
    
    DenoiseSnP();
    
}
Mat makeBinary(Mat img)
{
   Mat d_image = Mat::zeros( image.size(), image.type() );

    for( int y = 0; y < img.rows; y++ )
    { for( int x = 0; x < img.cols; x++ )
    {
        float r = (img.at<Vec3b>(y,x)[2]);
        float g = (img.at<Vec3b>(y,x)[1]);
        float b = (img.at<Vec3b>(y,x)[0]);
        
        if((r-g>50 && r-b>50) || (g>170 && b<200 )){
       // if(r>0){
            r = 255; g=255; b=255;      //white
        }
        else{
            r = 0; g = 0; b = 0;        //black
        }
        
        d_image.at<Vec3b>(y,x)[0] =
        saturate_cast<uchar>( b);
        d_image.at<Vec3b>(y,x)[1] =
        saturate_cast<uchar>( g );
        d_image.at<Vec3b>(y,x)[2] =
        saturate_cast<uchar>( r);
    }
    }
    
    
    int erosion_size = 6;
    Mat element = getStructuringElement(cv::MORPH_CROSS,
                                        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        cv::Point(erosion_size, erosion_size) );
    
    // Apply erosion or dilation on the image
    erode(d_image,d_image,element);
    medianBlur(d_image, d_image, 17);                   //to filter noise
    dilate(d_image, d_image, Mat(), Point(-1, -1), 2, 1, 1);
    
    int i=0,width=0,X=0,Y=0;
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            float r = (image.at<Vec3b>(y,x)[2]);
            
            if(r==255)
            {  i++;}
            else
            {
                if(i>width)
                {
                    X=x;
                    Y=y;
                    width = i;
                    
                }i=0;
            }
        }i=0;
    }X-=width/2;
    
    return d_image;
//    namedWindow("Binary", 1);
//    
//    imshow("Binary", d_image);
}
int main(int argc, const char * argv[]) {
        
        string imageName("/Users/melvinjames/Desktop/Apples.png");
        if( argc > 1)
        {
            imageName = argv[1];
        }
        
    
        image = imread(imageName.c_str(), IMREAD_COLOR); // Read the file
        temp =imread(imageName.c_str(), IMREAD_COLOR);
        
        if( image.empty() )                      // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
        imshow( "Display window", image );
    
        // For Gaussian Noise
        GaussN();
        //
    
        //For Salt and Pepper Noise
        SaltnPepper();
    
        // For adjusting Brightness
        B1 = Mat::zeros( image.size(), image.type() );
        for( int y = 0; y < image.rows; y++ )
        {
            for( int x = 0; x < image.cols; x++ )
            {
                for( int c = 0; c < 3; c++ )
                {
                    B1.at<Vec3b>(y,x)[c] =
                    saturate_cast<uchar>( image.at<Vec3b>(y,x)[c]  + 50 );
                }
            }
        }
        namedWindow("Brightness Adjusted Image B1", 1);
        
        imshow("Brightness Adjusted Image B1", B1);
        //
    
    Mat bin1 = makeBinary(image);
    //imshow("Binary of original", bin1);
    Mat bin2 = makeBinary(N1);
    //imshow("Binary of Gauss Noise N1", bin2);
    Mat bin3 = makeBinary(H1);
    //imshow("Binary of W/O Gauss Noise H1", bin3);
    Mat bin4 = makeBinary(N2);
    //imshow("Binary of SaltPepper N2", bin4);
    Mat bin5 = makeBinary(H2);
    //imshow("Binary of W/O Salt Pepper H2", bin5);
    Mat bin6 = makeBinary(B1);
    //imshow("Binary of Brightness Adjusted B1", bin6);
    
    cvtColor(bin1, bin1, CV_RGB2GRAY);
    cout<<"I ="<<connectedComponents(bin1,image, 8)<<endl;
    cvtColor(bin2, bin2, CV_RGB2GRAY);
    cout<<"N1 ="<<connectedComponents(bin2,image, 8)<<endl;
    cvtColor(bin3, bin3, CV_RGB2GRAY);
    cout<<"H1 ="<<connectedComponents(bin3,image, 8)<<endl;
    cvtColor(bin4, bin4, CV_RGB2GRAY);
    cout<<"N2 ="<<connectedComponents(bin4,image, 8)<<endl;
    cvtColor(bin5, bin5, CV_RGB2GRAY);
    cout<<"H2 ="<<connectedComponents(bin5,image, 8)<<endl;
    cvtColor(bin6, bin6, CV_RGB2GRAY);
    cout<<"B1 ="<<connectedComponents(bin6,image, 8)<<endl;

    waitKey(0);                         //to view output
    return 0;
}

# GFPGAN Raspberry Pi 4
![output image]( https://qengineering.eu/github/ManGAN.webp )
## Face reconstruction with the ncnn framework. <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/><br/>
Paper: https://arxiv.org/pdf/2101.04061.pdf<br/><br/>
Special made for a bare Raspberry Pi 4, see [Q-engineering deep learning examples](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html)

------------

## Dependencies.
To run the application, you have to:
- A raspberry Pi 4 with a 32 or 64-bit operating system. It can be the Raspberry 64-bit OS, or Ubuntu 18.04 / 20.04. [Install 64-bit OS](https://qengineering.eu/install-raspberry-64-os.html) <br/>
- The latest version of Tencent ncnn framework installed. [Install ncnn](https://qengineering.eu/install-ncnn-on-raspberry-pi-4.html) <br/>
- OpenCV 64 bit installed. [Install OpenCV 4.5](https://qengineering.eu/install-opencv-4.5-on-raspberry-64-os.html) <br/>
- Code::Blocks installed. (```$ sudo apt-get install codeblocks```)
- :point_right: Download and unzip the deep learning models `gfpgan-ncnn-20221225T122514Z-001.zip` file from [Gdrive](https://drive.google.com/file/d/1Lfs2fBU1ecaIKiQtMTaZW4q099PgPpA9/view?usp=share_link). 

------------

## Installing the app.
To extract and run the network in Code::Blocks <br/>
$ mkdir *MyDir* <br/>
$ cd *MyDir* <br/>
$ wget https://github.com/Qengineering/GFPGAN-ncnn-Raspberry-Pi-4/archive/refs/heads/main.zip <br/>
$ unzip -j master.zip <br/>
Remove master.zip, LICENSE and README.md as they are no longer needed. <br/> 
$ rm master.zip <br/>
$ rm LICENSE <br/>
$ rm README.md <br/> <br/>
Your *MyDir* folder must now look like this: <br/> 
```
.
├── Duo.png
├── Girl.jpg
├── Julia.jpg
├── Man.png
├── GFPGAN.cbp
├── include
│   ├── face.h
│   ├── gfpgan.h
│   └── realesrgan.h
├── models
│   ├── encoder.bin
│   ├── encoder.param
│   ├── real_esrgan.bin
│   ├── real_esrgan.param
│   ├── style.bin
│   ├── yolov5-blazeface.bin
│   └── yolov5-blazeface.param
└── src
    ├── face.cpp
    ├── gfpgan.cpp
    ├── main.cpp
    └── realesrgan.cpp
```
------------

## Running the app.
To run the application, load the GFPGAN.cbp project file into Code::Blocks. More information? Follow the instructions at [Hands-On](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html#HandsOn).<br/><br/>
In main.cpp, you have two branches: one for images with only one face (Girl.jpg) and another for scenes with more faces (Duo.png). The software determines which branch to follow. Especially the second branch, the one with the reconstruction of the whole scene, can take quite a long time. Mostly more than a few minutes.<br><br>
For best results, do not use jpeg compressed images. Strong jpeg compression generates typical artefacts to which the super-resolution algorithm does not respond well.<br/><br/>
![output image]( https://qengineering.eu/github/JuliaGAN.webp )
_(colorization done by https://github.com/Qengineering/ncnn-Colorization_Raspberry-Pi-4)_<br><br>
![output image]( https://qengineering.eu/github/GirlGAN.webp )<br><br>
![output image]( https://qengineering.eu/github/DuoGAN.webp )


------------

### Thanks.
A more than special thanks to [***FeiGeChuanShu***](https://github.com/FeiGeChuanShu), who adapted the ncnn framework for this app.<br>

------------

### More info.
Colorful Image Colorization [FeiGeChuanShu](https://github.com/FeiGeChuanShu/GFPGAN-ncnn)<br>
Colorful Image Colorization [Project Page](https://xinntao.github.io/projects/gfpgan) by Xintao Wang, Yu Li, Honglun Zhang, Ying Shan.

------------

[![paypal](https://qengineering.eu/images/TipJarSmall4.png)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CPZTM5BB3FCYL) 



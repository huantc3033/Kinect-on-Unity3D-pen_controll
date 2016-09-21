using UnityEngine;
using System;
using System.Collections;
using System.Runtime.InteropServices;
using System.Linq;
using Windows.Kinect;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;
using OpenCvSharp.Blob;

public class Pen_Controller : MonoBehaviour {

    GameObject pen;

    public int ColorWidth { get; private set; }
    public int ColorHeight { get; private set; }
    private KinectSensor _Sensor;
    private ColorFrameReader _ReaderColor;
    private DepthFrameReader _ReaderDepth;
    private Texture2D _TextureColor;
    private byte[] _DataColor;
    private System.IntPtr _IntPtrColor;
    private ushort[] _DataDepth;

    // OpenCV紀錄的Color和Depth影像
    Mat _ImgColor, _ImgColor_Bgr, _ImgColor_Hsv, _ImgRedMask, _ImgBlueMask;
    Mat _ImgDepth;
    DepthSpacePoint[] _ImgDepthPoints;

    Vector2 redCenter, blueCenter, lastredCenter, lastblueCenter, nextredCenter, nextblueCenter;
    // (new) 用來測Center位置是否有Update的變數
    private bool isRedCenterUpdated, isBlueCenterUpdated;
    // 紀錄包含深度的座標
    Vector3 redCenterCoordinate, blueCenterCoordinate;

    // (new) 轉換為真實世界座標用
    static readonly double HorizontalTanA = Math.Tan(70.0 / 2.0 * Math.PI / 180);
    static readonly double VerticalTanA = Math.Abs(Math.Tan(60.0 / 2.0 * Math.PI / 180));

    // 這裡用unity engine的固有Vector，比較好轉換
    Vector3 directionVector, positionVector, realdirectionVector, realpositionVector, lastpositionVector, lastdirectionVector;

    // Kalman Filter 的定義
    KalmanFilter positionFilter, directionFilter;
    Mat positionMeasurement, directionMeasurement;
    bool initializedpositionKalman = false, initializeddirectionKalman = false;
    bool redOnly = false, blueOnly = false;
    
	System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();

    // Use this for initialization
    void Start () {
        _Sensor = KinectSensor.GetDefault();

        if (_Sensor != null)
        {
            _ReaderColor = _Sensor.ColorFrameSource.OpenReader();

            var frameDesc = _Sensor.ColorFrameSource.CreateFrameDescription(ColorImageFormat.Rgba);
            //var frameDescDepth = _Sensor.DepthFrameSource.FrameDescription;
            ColorWidth = frameDesc.Width;
            ColorHeight = frameDesc.Height;

            _TextureColor = new Texture2D(frameDesc.Width, frameDesc.Height, TextureFormat.RGBA32, false);
            _DataColor = new byte[frameDesc.BytesPerPixel * frameDesc.LengthInPixels];

            _ReaderDepth = _Sensor.DepthFrameSource.OpenReader();
            _DataDepth = new ushort[_Sensor.DepthFrameSource.FrameDescription.LengthInPixels];

            if (!_Sensor.IsOpen)
            {
                _Sensor.Open();
            }

            _ImgColor = new Mat(frameDesc.Height, frameDesc.Width, MatType.CV_8UC4);
            _ImgColor_Bgr = new Mat(frameDesc.Height, frameDesc.Width, MatType.CV_8UC3);
            _ImgColor_Hsv = new Mat(frameDesc.Height, frameDesc.Width, MatType.CV_8UC3);
            _ImgDepth = new Mat(_Sensor.DepthFrameSource.FrameDescription.Height, _Sensor.DepthFrameSource.FrameDescription.Width, MatType.CV_16SC1);

            _ImgDepthPoints = new DepthSpacePoint[frameDesc.Height * frameDesc.Width];

            redCenter = new Vector2(-1, -1);
            blueCenter = new Vector2(-1, -1);
            lastredCenter = new Vector2(-1, -1);
            lastblueCenter = new Vector2(-1, -1);
            nextredCenter = new Vector2(-1, -1);
            nextblueCenter = new Vector2(-1, -1);

            isRedCenterUpdated = false;
            isBlueCenterUpdated = false;

            redCenterCoordinate = new Vector3(-1, -1, -1);
            blueCenterCoordinate = new Vector3(-1, -1, -1);

            positionVector = new Vector3(0, 0, 0);
            directionVector = new Vector3(0, 0, 0);
            realdirectionVector = new Vector3(-1, -1, -1);
            realpositionVector = new Vector3(-1, -1, -1);
			lastpositionVector = new Vector3 (-1, -1, -1);
			lastdirectionVector = new Vector3(-1, -1, -1);

            positionFilter = new KalmanFilter(9, 3, 0);
            positionMeasurement = new Mat(3, 1, MatType.CV_32FC1);
            positionMeasurement.SetTo(new Scalar(0), null);
			float [] posTransMat = { 1, 0, 0, 1, 0, 0, 0.5f, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.5f, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.5f, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
									0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
            positionFilter.TransitionMatrix.SetArray(0, 0, posTransMat);
            positionFilter.StatePre.Set<float>(0, 0);
            positionFilter.StatePre.Set<float>(1, 0);
            positionFilter.StatePre.Set<float>(2, 0);
            positionFilter.StatePre.Set<float>(3, 0);
            positionFilter.StatePre.Set<float>(4, 0);
            positionFilter.StatePre.Set<float>(5, 0);
			positionFilter.StatePre.Set<float>(6, 0);
			positionFilter.StatePre.Set<float>(7, 0);
			positionFilter.StatePre.Set<float>(8, 0);
            Cv2.SetIdentity(positionFilter.MeasurementMatrix);
            Cv2.SetIdentity(positionFilter.ProcessNoiseCov, new Scalar(1e-4));
            Cv2.SetIdentity(positionFilter.MeasurementNoiseCov, new Scalar(10));
            Cv2.SetIdentity(positionFilter.MeasurementMatrix, new Scalar(.99));

			float [] dirTransMat = { 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};

            directionFilter = new KalmanFilter(9, 3, 0);
            directionMeasurement = new Mat(3, 1, MatType.CV_32FC1);
            directionMeasurement.SetTo(new Scalar(0), null);
            directionFilter.TransitionMatrix.SetArray(0, 0, posTransMat);
            directionFilter.StatePre.Set<float>(0, 0);
            directionFilter.StatePre.Set<float>(1, 0);
            directionFilter.StatePre.Set<float>(2, 0);
            directionFilter.StatePre.Set<float>(3, 0);
            directionFilter.StatePre.Set<float>(4, 0);
            directionFilter.StatePre.Set<float>(5, 0);
            Cv2.SetIdentity(directionFilter.MeasurementMatrix);
            Cv2.SetIdentity(directionFilter.ProcessNoiseCov, new Scalar(1e-2));
            Cv2.SetIdentity(directionFilter.MeasurementNoiseCov, new Scalar(10));
            Cv2.SetIdentity(directionFilter.MeasurementMatrix, new Scalar(.99));
        }
    }

    // Update is called once per frame
    void Update()
    {
		stopwatch.Start ();

        if (_ReaderColor != null)
        {
            var frame = _ReaderColor.AcquireLatestFrame();

            if (frame != null)
            {
                frame.CopyConvertedFrameDataToArray(_DataColor, ColorImageFormat.Rgba);
                _TextureColor.LoadRawTextureData(_DataColor);
                _TextureColor.Apply();

                frame.CopyConvertedFrameDataToArray(_DataColor, ColorImageFormat.Bgra);
                ///////////////////////////////////////////////////////////////////////////////////////////
                Marshal.Copy(_DataColor, 0, _ImgColor.Data, _TextureColor.height * _TextureColor.width * 4);
                ///////////////////////////////////////////////////////////////////////////////////////////
                //frame.CopyConvertedFrameDataToIntPtr(_ImgColor.Data, (uint)(_ImgColor.Total() * _ImgColor.ElemSize()) , ColorImageFormat.Bgra);

                frame.Dispose();
                frame = null;
            }

        }

        // 讀取Depth Image
        if (_ReaderDepth != null)
        {
            var frame = _ReaderDepth.AcquireLatestFrame();
            if (frame != null)
            {
                frame.CopyFrameDataToArray(_DataDepth);
                //Debug.Log(_DataDepth[1560]);
                short[] _DataDepthS = Array.ConvertAll(_DataDepth, element => (short)element);
                Marshal.Copy(_DataDepthS, 0, _ImgDepth.Data, _ImgDepth.Height * _ImgDepth.Width);
                frame.Dispose();
                frame = null;
            }
        }

        Cv2.CvtColor(_ImgColor, _ImgColor_Bgr, ColorConversion.BgraToBgr);
        Cv2.CvtColor(_ImgColor_Bgr, _ImgColor_Hsv, ColorConversion.BgrToLab);

        _Sensor.CoordinateMapper.MapColorFrameToDepthSpace(_DataDepth, _ImgDepthPoints);
        //_ImgDepthMap = new Mat(_ImgColor.Height, _ImgColor.Width, MatType.CV_32SC2, _ImgDepthPoints);
        //Debug.Log(_ImgDepthMap.Get<Vec2s>(100, 0)[0] + "," + _ImgDepthMap.Get<Vec2s>(100, 0)[1]);

		stopwatch.Stop ();

		//  获取当前实例测量得出的总时间
		System.TimeSpan timespan = stopwatch.Elapsed;
		//   double hours = timespan.TotalHours; // 總小時
		//    double minutes = timespan.TotalMinutes;  // 總分鐘
		//    double seconds = timespan.TotalSeconds;  //  總秒數
		double milliseconds1 = timespan.TotalMilliseconds;  //  總毫秒數



        // 初始化之後要用的Moments
        Moments _imgMomentsR;
        Moments _imgMomentsB;
        double dM01R;
        double dM10R;
        double dAreaR;
        double dM01B;
        double dM10B;
        double dAreaB;

        isRedCenterUpdated = false;
        isBlueCenterUpdated = false;

		stopwatch.Reset ();
		stopwatch.Start ();

        // 算中心點的區塊(紅色部分)
        // (更新)一開始先算全部的中心點，之後直接以上一frame的速度預測下一frame的位置
        // 在附近形成ROI，在ROI裡的區塊以較寬的條件夾擊
        // 如果找不到，再退回用全部的區塊算
        // 若再找不到，則用(-1, -1)

        // 如果nextredCenter不為(-1, -1)，找ROI的位置
        if (nextredCenter != new Vector2(-1, -1))
        {
            if (nextredCenter.x >= 50 && nextredCenter.x < 1870 && nextredCenter.y >= 50 && nextredCenter.y < 1030)
            {
                // 預測位置存在，開始以此展開ROI
                OpenCvSharp.CPlusPlus.Rect regionR = new OpenCvSharp.CPlusPlus.Rect(new Point(nextredCenter.x - 50, nextredCenter.y - 50), new Size(100, 100));
                //Debug.Log(nextredCenter);
                Mat ROI = _ImgColor_Hsv.SubMat(regionR);
                Mat _ImgRedMaskROI = new Mat(ROI.Size(), MatType.CV_8UC3);

                Cv2.InRange(ROI, new Scalar(0, 0, 127), new Scalar(255, 119, 150), _ImgRedMaskROI);
                Cv2.Erode(_ImgRedMaskROI, _ImgRedMaskROI, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
                Cv2.Dilate(_ImgRedMaskROI, _ImgRedMaskROI, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));

                Cv2.ImShow("RedMaskROI", _ImgRedMaskROI);

                _imgMomentsR = new Moments(_ImgRedMaskROI);
                dM01R = _imgMomentsR.M01;
                dM10R = _imgMomentsR.M10;
                dAreaR = _imgMomentsR.M00;
                if (dAreaR > 50)
                {
                    //calculate the position of the ball
                    int posXR = (int)(dM10R / dAreaR);
                    int posYR = (int)(dM01R / dAreaR);

                    // 現在的redCenter成為之前的
                    lastredCenter = redCenter;
                    // 存最新的位置
                    redCenter = new Vector2(nextredCenter.x + posXR - 50, nextredCenter.y + posYR - 50); // 這是對於Frame格式來說的XY，也就是(col, row)
                    //Debug.Log("Current:" + redCenter);
                    nextredCenter = redCenter;
                    isRedCenterUpdated = true;
                }
                else
                {
                    // 沒有找到，不能update center位置，只能等之後全域搜尋
                    isRedCenterUpdated = false;
                }
            }
            else
            {
                isRedCenterUpdated = false;
            }
        }

        if (!isRedCenterUpdated)
        {
            if (nextredCenter.x >= 50 && nextredCenter.x < 1870 && nextredCenter.y >= 50 && nextredCenter.y < 1030)
            {
                // 預測位置存在，開始以此展開ROI
                OpenCvSharp.CPlusPlus.Rect regionR = new OpenCvSharp.CPlusPlus.Rect(new Point(nextredCenter.x - 50, nextredCenter.y - 50), new Size(100, 100));
                //Debug.Log(nextredCenter);
                Mat ROI = _ImgColor_Hsv.SubMat(regionR);
                Mat _ImgRedMaskROI = new Mat(ROI.Size(), MatType.CV_8UC3);

                Cv2.InRange(ROI, new Scalar(0, 0, 130), new Scalar(255, 117, 146), _ImgRedMaskROI);
                Cv2.Erode(_ImgRedMaskROI, _ImgRedMaskROI, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
                Cv2.Dilate(_ImgRedMaskROI, _ImgRedMaskROI, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));

                Cv2.ImShow("RedMaskROI", _ImgRedMaskROI);

                _imgMomentsR = new Moments(_ImgRedMaskROI);
                dM01R = _imgMomentsR.M01;
                dM10R = _imgMomentsR.M10;
                dAreaR = _imgMomentsR.M00;
                if (dAreaR > 50)
                {
                    //calculate the position of the ball
                    int posXR = (int)(dM10R / dAreaR);
                    int posYR = (int)(dM01R / dAreaR);

                    // 現在的redCenter成為之前的
                    lastredCenter = redCenter;
                    // 存最新的位置
                    redCenter = new Vector2(nextredCenter.x + posXR - 50, nextredCenter.y + posYR - 50); // 這是對於Frame格式來說的XY，也就是(col, row)
                    //Debug.Log("Current:" + redCenter);
                    nextredCenter = redCenter;
                    isRedCenterUpdated = true;
                }
                else
                {
                    // 沒有找到，不能update center位置，只能等之後全域搜尋
                    isRedCenterUpdated = false;
                }
            }
            else
            {
                isRedCenterUpdated = false;
            }
        }

        // 初始時、沒有預測下一個點的位置或ROI位置搜尋失敗，開始全域搜尋，此時尋找值範圍縮小
        if (!isRedCenterUpdated)
        {
            _ImgRedMask = new Mat(_ImgColor.Size(), MatType.CV_8UC3);
            Cv2.InRange(_ImgColor_Hsv, new Scalar(0, 0, 101), new Scalar(114, 114, 255), _ImgRedMask);
            Cv2.Erode(_ImgRedMask, _ImgRedMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
            Cv2.Dilate(_ImgRedMask, _ImgRedMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));

            _imgMomentsR = new Moments(_ImgRedMask);
            dM01R = _imgMomentsR.M01;
            dM10R = _imgMomentsR.M10;
            dAreaR = _imgMomentsR.M00;

            if (dAreaR > 50)
            {
                //calculate the position of the ball
                int posXR = (int)(dM10R / dAreaR);
                int posYR = (int)(dM01R / dAreaR);

                lastredCenter = redCenter;
                redCenter = new Vector2(posXR, posYR); // 這是對於Frame格式來說的XY，也就是(col, row)
                //Debug.Log("Current:" + redCenter);
                // 這代表上次計算中心也是成功的，這樣就可以獲得預測值
                if (lastredCenter != new Vector2(-1, -1) && lastredCenter.x >= 0 && lastredCenter.x < 1920 && lastredCenter.y >= 0 && lastredCenter.y < 1080)
                {
                    nextredCenter = redCenter;
                }
                else
                {
                    nextredCenter = new Vector2(-1, -1);
                }
                isRedCenterUpdated = true;
            }
            else
            {
                // 全域搜尋也找不到，設Center為(-1, -1)
                // (之後再做)如果前一個位置已知，那可以透過該地延伸的roi獲得位置。風險是因為值的範圍擴大，會掃到無關的東西
                lastredCenter = redCenter;
                redCenter = new Vector2(-1, -1);
                //Debug.Log("Current:" + redCenter);
                nextredCenter = new Vector2(-1, -1);
                isRedCenterUpdated = true;
            }
        }

        // 算區塊的中心點(藍色部分)
        // 如果nextblueCenter不為(-1, -1)，找ROI的位置
        if (nextblueCenter != new Vector2(-1, -1) && nextblueCenter.x >= 0 && nextblueCenter.x < 1920 && nextblueCenter.y >= 0 && nextblueCenter.y < 1080)
        {
            if (nextblueCenter.x >= 50 && nextblueCenter.x < 1870 && nextblueCenter.y >= 50 && nextblueCenter.y < 1030)
            {
                // 預測位置存在，開始以此展開ROI
                OpenCvSharp.CPlusPlus.Rect regionB = new OpenCvSharp.CPlusPlus.Rect(new Point(nextblueCenter.x - 50, nextblueCenter.y - 50), new Size(100, 100));
                Mat ROI = _ImgColor_Hsv.SubMat(regionB);
                Mat _ImgBlueMaskROI = new Mat(ROI.Size(), MatType.CV_8UC3);

                Cv2.InRange(ROI, new Scalar(63, 0, 172), new Scalar(255, 255, 255), _ImgBlueMaskROI);
                Cv2.Erode(_ImgBlueMaskROI, _ImgBlueMaskROI, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
                Cv2.Dilate(_ImgBlueMaskROI, _ImgBlueMaskROI, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));

                Cv2.ImShow("BlueMaskROI", _ImgBlueMaskROI);

                _imgMomentsB = new Moments(_ImgBlueMaskROI);
                dM01B = _imgMomentsB.M01;
                dM10B = _imgMomentsB.M10;
                dAreaB = _imgMomentsB.M00;
                if (dAreaB > 50)
                {
                    //calculate the position of the ball
                    int posXB = (int)(dM10B / dAreaB);
                    int posYB = (int)(dM01B / dAreaB);

                    // 現在的blueCenter成為之前的
                    lastredCenter = blueCenter;
                    // 存最新的位置
                    blueCenter = new Vector2(nextblueCenter.x + posXB - 50, nextblueCenter.y + posYB - 50); // 這是對於Frame格式來說的XY，也就是(col, row)
                    nextblueCenter = blueCenter;
                    isBlueCenterUpdated = true;
                }
                else
                {
                    // 沒有找到，不能update center位置，只能等之後全域搜尋
                    isBlueCenterUpdated = false;
                }
            }
            else
            {
                isBlueCenterUpdated = false;
            }
        }

        // 初始時、沒有預測下一個點的位置或ROI位置搜尋失敗，開始全域搜尋，此時尋找值範圍縮小
        if (!isBlueCenterUpdated)
        {
            _ImgBlueMask = new Mat(_ImgColor.Size(), MatType.CV_8UC3);
            Cv2.InRange(_ImgColor_Hsv, new Scalar(103, 0, 174), new Scalar(255, 127, 255), _ImgBlueMask);
            //morphological opening (removes small objects from the foreground)
            Cv2.Erode(_ImgBlueMask, _ImgBlueMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
            Cv2.Dilate(_ImgBlueMask, _ImgBlueMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
            //morphological closing (removes small holes from the foreground)
            //Cv2.Dilate(_ImgRedMask, _ImgRedMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
            //Cv2.Dilate(_ImgBlueMask, _ImgBlueMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
            //Cv2.Erode(_ImgRedMask, _ImgRedMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));
            //Cv2.Erode(_ImgBlueMask, _ImgBlueMask, Cv2.GetStructuringElement(StructuringElementShape.Ellipse, new Size(5, 5)));

            _imgMomentsB = new Moments(_ImgBlueMask);
            dM01B = _imgMomentsB.M01;
            dM10B = _imgMomentsB.M10;
            dAreaB = _imgMomentsB.M00;
            if (dAreaB > 50)
            {
                int posXB = (int)(dM10B / dAreaB);
                int posYB = (int)(dM01B / dAreaB);

                lastblueCenter = blueCenter;
                blueCenter = new Vector2(posXB, posYB); // 這是對於Frame格式來說的XY，也就是(col, row)
                if (lastblueCenter != new Vector2(-1, -1))
                {
                    nextblueCenter = blueCenter;
                }
                else
                {
                    nextblueCenter = new Vector2(-1, -1);
                }
                isBlueCenterUpdated = true;
            }
            else
            {
                lastblueCenter = blueCenter;
                blueCenter = new Vector2(-1, -1); // 沒有中心，設為(-1, -1)
                nextblueCenter = new Vector2(-1, -1);
                isBlueCenterUpdated = true;
            }
        }

		stopwatch.Stop ();

		//  获取当前实例测量得出的总时间
		System.TimeSpan timespan2 = stopwatch.Elapsed;
		//   double hours = timespan2.TotalHours; // 總小時
		//    double minutes = timespan2.TotalMinutes;  // 總分鐘
		//    double seconds = timespan2.TotalSeconds;  //  總秒數
		double milliseconds2 = timespan2.TotalMilliseconds;  //  總毫秒數

        /////////////////////////////////////////////////////////////////////////
        /// 補充：在存img等資料格式時，會習慣用(row, col)的方式來存
        /// 故在影像的1維陣列中要找特定像素的資料，需要用[row * numOfCol + col]的方式來找
        /// 但在(x, y)坐標系裡，則是先(col)再(row)
        /// 所以用x, y座標找1維陣列時需要用[y * numOfX + x]的方式來尋找
        /// 而在引用座標時，要直接用(x, y)(尤其在畫圓時)
        /////////////////////////////////////////////////////////////////////////

        // 測試用程式碼(可取消註解來測試)
        //Cv2.Circle(_ImgColor, new Point2f(blueCenter.x, blueCenter.y), 10, new Scalar(0, 0, 255));
        //Cv2.ImShow("Img", _ImgColor);
        //Cv2.ImShow("RedMask", _ImgRedMask);
        //Cv2.ImShow("BlueMask", _ImgBlueMask);

		stopwatch.Reset ();
		stopwatch.Start ();

        // 找中心點的深度值，如果(1)沒有測到中心點或(2)其深度位置為(-infinity, -infinity)(代表沒有mapping到此處)則跳過
        // (待做)如果深度值非期望，則找附近的值
        // (待做)如果只有一個區域有中心點，則設定官方direction vector
        if (redCenter.x >= 50 && blueCenter.x >= 50 && redCenter.y >= 50 && blueCenter.y >= 50 && redCenter.x < 1870 && blueCenter.x < 1870 && redCenter.y < 1030 && blueCenter.y < 1030)
        {
            //Debug.Log(redCenter);
            // 紅(綠)色部分。其深度位置為(-infinity, -infinity)(代表沒有mapping到此處)則跳過
            if (_ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].X != float.NegativeInfinity && _ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].Y != float.NegativeInfinity
                && _ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].X != -2147483648 && _ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].Y != -2147483648)
            {
                int z = _ImgDepth.Get<short>((int)(_ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].Y), (int)(_ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].X));

                z = z >> 3;

                redCenterCoordinate = new Vector3(redCenter.x, redCenter.y, z);
                //Debug.Log(redCenterCoordinate);
            }
            else
            {
                // 此向量會讓接下來的方向向量/位置向量處理器維持之前的向量
                redCenterCoordinate = new Vector3(-1, 0, 0);
            }
            // 藍色部分。其深度位置為(-infinity, -infinity)(代表沒有mapping到此處)則跳過
            if (_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X != float.NegativeInfinity && _ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y != float.NegativeInfinity
                && _ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X != -2147483648 && _ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y != -2147483648)
            {
                //Debug.Log((int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X) + ", " + (int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y));
                int z = _ImgDepth.Get<short>((int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y), (int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X));
                z = z >> 3;
                blueCenterCoordinate = new Vector3(blueCenter.x, blueCenter.y, z);
            }
            else
            {
                // 此向量會讓接下來的方向向量/位置向量處理器維持之前的向量
                blueCenterCoordinate = new Vector3(-2, 0, 0);
            }
        }
        else if (redCenter.x >= 0 && redCenter.y >= 0 && redCenter.x < 1870 && redCenter.y < 1030)
        {
            // 紅色面向螢幕
            if (_ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].X != float.NegativeInfinity && _ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].Y != float.NegativeInfinity
                && _ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].X != -2147483648 && _ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].Y != -2147483648)
            {
                int z = _ImgDepth.Get<short>((int)(_ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].Y), (int)(_ImgDepthPoints[(int)(redCenter.y * 1920 + redCenter.x)].X));

                z = z >> 3;

                redCenterCoordinate = new Vector3(redCenter.x, redCenter.y, z);
                //Debug.Log(redCenterCoordinate);
            }
            else
            {
                // 此向量會讓接下來的方向向量/位置向量處理器維持之前的向量
                redCenterCoordinate = new Vector3(-1, 0, 0);
            }

            blueCenterCoordinate = new Vector3(-2, 0, 0);
        }
        else if (blueCenter.x >= 0 && blueCenter.y >= 0 && blueCenter.x < 1870 && blueCenter.y < 1030)
        {
            // 藍色面向螢幕
            if (_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X != float.NegativeInfinity && _ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y != float.NegativeInfinity
                && _ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X != -2147483648 && _ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y != -2147483648)
            {
                //Debug.Log((int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X) + ", " + (int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y));
                int z = _ImgDepth.Get<short>((int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].Y), (int)(_ImgDepthPoints[(int)(blueCenter.y * 1920 + blueCenter.x)].X));
                z = z >> 3;
                blueCenterCoordinate = new Vector3(blueCenter.x, blueCenter.y, z);
            }
            else
            {
                // 此向量會讓接下來的方向向量/位置向量處理器維持之前的向量
                blueCenterCoordinate = new Vector3(-2, 0, 0);
            }

            redCenterCoordinate = new Vector3(-1, 0, 0);
        }
        else
        {
            // 不做任何事
            // 此向量會讓接下來的方向向量/位置向量處理器維持之前的向量
            redCenterCoordinate = new Vector3(-1, 0, 0);
            blueCenterCoordinate = new Vector3(-2, 0, 0);
        }

		stopwatch.Stop ();

		//  获取当前实例测量得出的总时间
		System.TimeSpan timespan3 = stopwatch.Elapsed;
		//   double hours = timespan3.TotalHours; // 總小時
		//    double minutes = timespan3.TotalMinutes;  // 總分鐘
		//    double seconds = timespan3.TotalSeconds;  //  總秒數
		double milliseconds3 = timespan3.TotalMilliseconds;  //  總毫秒數

		stopwatch.Reset ();
		stopwatch.Start ();

        // 這裡是方向向量/位置向量處理器
        // 換算成真實世界的coordinate，計算方向與位置向量
        if (!redCenterCoordinate.Equals(new Vector3(-1, 0, 0)) && !blueCenterCoordinate.Equals(new Vector3(-2, 0, 0)))
        {
            // 利用深度值換成真實世界的座標
            // 定義六個真實世界的座標值
            int realRedX, realRedY, realRedZ;
            int realBlueX, realBlueY, realBlueZ;

            // 定義opposite，real座標 = (pixel座標-一半的寬/高) * 2b / 寬/高
            float redOppositeX = (float)redCenterCoordinate.z * (float)HorizontalTanA;
            float redOppositeY = (float)redCenterCoordinate.z * (float)VerticalTanA;
            float blueOppositeX = (float)blueCenterCoordinate.z * (float)HorizontalTanA;
            float blueOppositeY = (float)blueCenterCoordinate.z * (float)VerticalTanA;

            // (X與Y部分)先導入depthImage的座標值
            realRedX = (int)(_ImgDepthPoints[(int)(redCenterCoordinate.y * 1920 + redCenterCoordinate.x)].X);
            realRedY = (int)(_ImgDepthPoints[(int)(redCenterCoordinate.y * 1920 + redCenterCoordinate.x)].Y);
            realBlueX = (int)(_ImgDepthPoints[(int)(blueCenterCoordinate.y * 1920 + blueCenterCoordinate.x)].X);
            realBlueY = (int)(_ImgDepthPoints[(int)(blueCenterCoordinate.y * 1920 + blueCenterCoordinate.x)].Y);

            // (X與Y部分)算此處的真實座標值
            realRedX = (int)((realRedX - 256) * 2 * redOppositeX / 512);
            realRedY = (int)((realRedY - 212) * 2 * redOppositeY / 424);
            realBlueX = (int)((realBlueX - 256) * 2 * blueOppositeX / 512);
            realBlueY = (int)((realBlueY - 212) * 2 * blueOppositeY / 424);

            // (Z部分)沿用之前的z數據
            realRedZ = (int)redCenterCoordinate.z;
            realBlueZ = (int)blueCenterCoordinate.z;

            realdirectionVector = new Vector3(realRedX - realBlueX, realRedY - realBlueY, realRedZ - realBlueZ);
			lastpositionVector = realpositionVector;
			realpositionVector = new Vector3((realBlueX), (realBlueY), (realBlueZ));

            //directionVector = new Vector3((redCenterCoordinate.x - blueCenterCoordinate.x), (redCenterCoordinate.y - blueCenterCoordinate.y), (redCenterCoordinate.z - blueCenterCoordinate.z));
            //positionVector = new Vector3((redCenterCoordinate.x + blueCenterCoordinate.x) / 2, (redCenterCoordinate.y + blueCenterCoordinate.y) / 2, (redCenterCoordinate.z + blueCenterCoordinate.z) / 2);

            // 為了平衡用
            redOnly = false;
            blueOnly = false;
            //Debug.Log(positionVector);
        }
        else if (!redCenterCoordinate.Equals(new Vector3(-1, 0, 0)))
        {
            //Debug.Log("2");
            // 藍色區域被擋住
            int realRedX, realRedY, realRedZ;
            float redOppositeX = (float)redCenterCoordinate.z * (float)HorizontalTanA;
            float redOppositeY = (float)redCenterCoordinate.z * (float)VerticalTanA;
            realRedX = (int)(_ImgDepthPoints[(int)(redCenterCoordinate.y * 1920 + redCenterCoordinate.x)].X);
            realRedY = (int)(_ImgDepthPoints[(int)(redCenterCoordinate.y * 1920 + redCenterCoordinate.x)].Y);
            realRedX = (int)((realRedX - 256) * 2 * redOppositeX / 512);
            realRedY = (int)((realRedY - 212) * 2 * redOppositeY / 424);
            realRedZ = (int)redCenterCoordinate.z;

			lastdirectionVector = realdirectionVector;
            realdirectionVector = new Vector3(0, 0, 10);
			lastpositionVector = realpositionVector;
            realpositionVector = new Vector3((realRedX), (realRedY), (realRedZ));

            redOnly = true;
            blueOnly = false;
        }
        else if (!blueCenterCoordinate.Equals(new Vector3(-2, 0, 0)))
        {
            //Debug.Log("1");
            // 紅(綠)色區域被擋住
            int realBlueX, realBlueY, realBlueZ;
            float blueOppositeX = (float)blueCenterCoordinate.z * (float)HorizontalTanA;
            float blueOppositeY = (float)blueCenterCoordinate.z * (float)VerticalTanA;
            realBlueX = (int)(_ImgDepthPoints[(int)(blueCenterCoordinate.y * 1920 + blueCenterCoordinate.x)].X);
            realBlueY = (int)(_ImgDepthPoints[(int)(blueCenterCoordinate.y * 1920 + blueCenterCoordinate.x)].Y);
            realBlueX = (int)((realBlueX - 256) * 2 * blueOppositeX / 512);
            realBlueY = (int)((realBlueY - 212) * 2 * blueOppositeY / 424);
            realBlueZ = (int)blueCenterCoordinate.z;

			lastdirectionVector = realdirectionVector;
            realdirectionVector = new Vector3(0, 0, -10);
			lastpositionVector = realpositionVector;
            realpositionVector = new Vector3((realBlueX), (realBlueY), (realBlueZ));

            redOnly = false;
            blueOnly = true;
        }
        else
        {
            // 偵測不到。不做任何事，維持之前的向量
        }

		stopwatch.Stop ();

		//  获取当前实例测量得出的总时间
		System.TimeSpan timespan4 = stopwatch.Elapsed;
		//   double hours = timespan4.TotalHours; // 總小時
		//    double minutes = timespan4.TotalMinutes;  // 總分鐘
		//    double seconds = timespan4.TotalSeconds;  //  總秒數
		double milliseconds4 = timespan4.TotalMilliseconds;  //  總毫秒數

		stopwatch.Reset ();
		stopwatch.Start ();

        // 以兩coordinate的中心為位置，兩coordinate的方向為方向，畫出筆
		// 使用Kalman Filter來平滑軌跡及預測位置

		// 定義筆的物件
        pen = GameObject.Find("Pen");
		// 若：
		// 		1. 前一個位置不為未輸入狀態(若有輸入則不可能為(-1, -1, -1))
		// 		2. 現位置z不大於150.0f(為防止誤算深度到背後的景物)
		// 		3. 現位置減前一個位置小於10.0f(進一步防止震動)
		// 		4. 已經將Kalman filter經過初始化
		// 此時照正常的步驟更新Kalman Filter，我們將現位置輸入更新，得到Filter過後的值
		if (!lastpositionVector.Equals (new Vector3 (-1, -1, -1)) && Mathf.Abs(realpositionVector.z) <= 150.0f && Mathf.Abs(realpositionVector.z-lastpositionVector.z) <= 10.0f  && initializedpositionKalman) {
			// 以Kalman Filter修正位置與方向向量
			// First predict, to update the internal statePre variable

			Mat positionPredicted = positionFilter.Predict ();
			Vector3 positionPredictedVector = new Vector3 (positionPredicted.Get<float> (0), positionPredicted.Get<float> (1), positionPredicted.Get<float> (2));

			// 測量值為我們給定的向量
			positionMeasurement.Set<float> (0, realpositionVector.x);
			positionMeasurement.Set<float> (1, realpositionVector.y);
			positionMeasurement.Set<float> (2, realpositionVector.z);
			Mat positionEstimated = positionFilter.Correct (positionMeasurement);
			realpositionVector = new Vector3 (positionEstimated.Get<float> (0), positionEstimated.Get<float> (1), positionEstimated.Get<float> (2));

			//Debug.Log (new Vector3 (positionEstimated.Get<float> (0), positionEstimated.Get<float> (1), positionEstimated.Get<float> (2)));

			//Debug.Log(realpositionVector);

			pen.transform.position = new Vector3 (realpositionVector.x, -realpositionVector.y, 150.0f - realpositionVector.z);
		} 
		// 若：
		// 		1. 前一個位置不為未輸入狀態(若有輸入則不可能為(-1, -1, -1))
		// 		2. 現位置z大於150.0f(處理過大的情況)
		// 		3. 已經將Kalman filter經過初始化
		// 此時不更新Kalman Filter，直接將預測值代入筆的位置
		else if ((!lastpositionVector.Equals (new Vector3 (-1, -1, -1)) && Mathf.Abs(realpositionVector.z) > 150.0f && initializedpositionKalman)) {
			Mat positionPredicted = positionFilter.Predict ();
			realpositionVector = new Vector3 (positionPredicted.Get<float> (0), positionPredicted.Get<float> (1), positionPredicted.Get<float> (2));

			//Debug.Log(realpositionVector);

			pen.transform.position = new Vector3 (realpositionVector.x, -realpositionVector.y, 150.0f - realpositionVector.z);
		} 
		// 若：
		// 		1. 前一個位置不為未輸入狀態(若有輸入則不可能為(-1, -1, -1))
		// 		2. 已經將Kalman filter經過初始化
		// 此時要先初始化Kalman Filter，並事先將其PreState和PostState調成初始位置。在此之前，先將筆的位置移動到初始位置
		// 尤其是PostState重要，如此可避免一開始位置出現在原點，需要等一段時間位置才會就定位的狀況
		else if (!lastpositionVector.Equals (new Vector3 (-1, -1, -1)) && !initializedpositionKalman)
		{
			pen.transform.position = new Vector3 (realpositionVector.x, -realpositionVector.y, 150.0f - realpositionVector.z);

			//Debug.Log (realpositionVector);

			if (!initializedpositionKalman) {
				// initialize 位置向量
				// 將Kalman Filter的計算前state預先設置為起始位置。只是為防萬一
				positionFilter.StatePre.Set<float> (0, realpositionVector.x);
				positionFilter.StatePre.Set<float> (1, realpositionVector.y);
				positionFilter.StatePre.Set<float> (2, realpositionVector.z);
				// (Key point)將Kalman Filter的計算後state預先設置為起始位置，可避免從原點飛出來的情況(2016/8/17)
				positionFilter.StatePost.Set<float> (0, realpositionVector.x);
				positionFilter.StatePost.Set<float> (1, realpositionVector.y);
				positionFilter.StatePost.Set<float> (2, realpositionVector.z);
				initializedpositionKalman = true;
			}
		}
		// 其餘情況，包括震動超過10.0f的部分
        else
        {
            // 不做任何事，維持之前的向量
        }

		/*if (!realdirectionVector.Equals(new Vector3(0, 0, 0)) && Mathf.Abs(realdirectionVector.z) <= 30.0f)
		{
			if (!initializeddirectionKalman)
			{
				directionFilter.StatePost.Set<float>(0, realdirectionVector.x);
				directionFilter.StatePost.Set<float>(1, realdirectionVector.y);
				directionFilter.StatePost.Set<float>(2, realdirectionVector.z);
				initializeddirectionKalman = true;
			}

			Mat directionPredicted = directionFilter.Predict();
			Vector3 directionPredictedVector = new Vector3(directionPredicted.Get<float>(0), directionPredicted.Get<float>(1), directionPredicted.Get<float>(2));

			if (redOnly)
			{
				realdirectionVector = new Vector3(directionPredicted.Get<float>(0), directionPredicted.Get<float>(1), directionPredicted.Get<float>(2));
				//realdirectionVector = lastdirectionVector;
			}
			else if (blueOnly)
			{
				realdirectionVector = new Vector3(directionPredicted.Get<float>(0), directionPredicted.Get<float>(1), directionPredicted.Get<float>(2));
				//realdirectionVector = lastdirectionVector;
			}
			else
			{
				directionMeasurement.Set<float>(0, realdirectionVector.x);
				directionMeasurement.Set<float>(1, realdirectionVector.y);
				directionMeasurement.Set<float>(2, realdirectionVector.z);
				Mat directionEstimated = directionFilter.Correct(directionMeasurement);
				realdirectionVector = new Vector3(directionEstimated.Get<float>(0), directionEstimated.Get<float>(1), directionEstimated.Get<float>(2));
			}

			pen.transform.rotation = Quaternion.FromToRotation(new Vector3(0, 1, 0), new Vector3(realdirectionVector.x, -realdirectionVector.y, -realdirectionVector.z));
			//Debug.Log("x: "+xAngle+"z: "+zAngle);

			//Debug.Log(realdirectionVector);
		}
		else
		{
			
		}*/

		stopwatch.Stop ();

		//  获取当前实例测量得出的总时间
		System.TimeSpan timespan5 = stopwatch.Elapsed;
		//   double hours = timespan5.TotalHours; // 總小時
		//    double minutes = timespan5.TotalMinutes;  // 總分鐘
		//    double seconds = timespan5.TotalSeconds;  //  總秒數
		double milliseconds5 = timespan5.TotalMilliseconds;  //  總毫秒數

		// 輸出code執行時間
		Debug.Log(milliseconds1 + " " + milliseconds2 + " " + milliseconds3 + " " + milliseconds4 + " " + milliseconds5);

		stopwatch.Reset ();
        // Cv2.ImShow("BlueMask", _ImgBlueMask);
        //Cv2.ImShow("RedMask", _ImgRedMask);
        // 開始Kinect Code
        /*for(int i = 0; i < _TextureColor.width; i++)
        //{
        //    for (int j = 0; j < _TextureColor.height; j++)
        //    {
        //        Color Pixel = _TextureColor.GetPixel(i, j);
        //        if (Pixel.r - Pixel.g > 0.3f && Pixel.r - Pixel.g > 0.3f)
        //        {
        //            redNumber++;
        //            redCenter += new Vector2(i, j);
        //        }
        //        else if (Pixel.b - Pixel.g > 0.3f && Pixel.b - Pixel.r > 0.3f)
        //        {
        //            blueNumber++;
        //            blueCenter += new Vector2(i, j);
        //        }
        //    }
        //}

        //redCenter /= redNumber;
        //blueCenter /= blueNumber;*/
    }
}

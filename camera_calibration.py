import numpy as np
import cv2 as cv
import datetime

TW = 800 # Target width for display
#is_paused = False
#is_selecting = False


def select_img_from_video(video_file, board_pattern, recorder, w, h, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    # Grab an images from the video
    valid, img = video.read()

    is_paused = False
    is_selecting = False

    while valid:
        # 프레임 크기 조절
        dim = (w, h)
        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

            if is_paused:             # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
            elif is_selecting:        # Enter: Select the image
                img_select.append(img)
                is_selecting = False

            cv.imshow(wnd_name, display)
            recorder.write(display) # Record the video with the display

        # 키 입력 처리
        key = cv.waitKey(wait_msec)
          
        if key == ord(' '):     # Space: Pause and show corners
            is_paused = not is_paused
        elif key == ord('\r'):  # Enter: Select the image
            if is_paused:
                is_selecting = not is_selecting
                is_paused = False # 선택 후 일시정지 해제
        elif key == 27:     # ESC: Exit (Complete image selection)
            break

        # 일시정지 상태가 아닐 때만 다음 프레임을 읽음
        if not is_paused:
            valid, img = video.read()

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

def distortion_correction(target_video, target_K, target_dist_coeff, recorder, result_recorder, w, h):
    # The given video and calibration data
    video_file = target_video
    K = target_K # Derived from `calibrate_camera.py`
    dist_coeff = target_dist_coeff # Derived from `calibrate_camera.py`

    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file

    # Run distortion correction
    show_rectify = True
    map1, map2 = None, None
    # 결과 영상 저장
    while True:
        valid, img = video.read()
        if not valid: 
            video = cv.VideoCapture(video_file)
            map1, map2 = None, None
            break
        
        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
        
        if map1 is None or map2 is None:
            map1, map2 = cv.initUndistortRectifyMap(target_K, target_dist_coeff, None, None, (w, h), cv.CV_32FC1)
        
        # 보정 수행
        rectified_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        
        # 파일에 결과 저장
        result_recorder.write(rectified_img)
    # 재생 및 비교
    while True:
        # Read an image from the video
        valid, img = video.read()

        if not valid:
            break
        dim = (w, h)
        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        
        # Rectify geometric distortion (Alternative: `cv.undistort()`)
        info = "Original"
        if show_rectify:
            if map1 is None or map2 is None:
                map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
            img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
            info = "Rectified"
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # Show the image and process the key event
        cv.imshow("Geometric Distortion Correction", img)
        recorder.write(img) # Record the video with the display
        key = cv.waitKey(10)
        if key == ord(' '):     # Space: Pause
            key = cv.waitKey()
        if key == 27:           # ESC: Exit
            break
        elif key == ord('\t'):  # Tab: Toggle the mode
            show_rectify = not show_rectify



if __name__ == '__main__':
    video_file = 'data.mp4'
    video_name = video_file.split('.')[0]
    board_pattern = (10, 7)
    board_cellsize = 0.025
    # 비디오 설정을 위한 정보 가져오기
    cap = cv.VideoCapture(video_file)
    fps = cap.get(cv.CAP_PROP_FPS) # 원본 FPS 가져오기
    width_original = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # 가로를 TW로 고정하고 세로 비율 유지
    target_width = TW
    aspect_ratio = height_original / width_original # 세로/가로 비율
    target_height = int(target_width * aspect_ratio)
    fourcc = cv.VideoWriter_fourcc(*'XVID') # AVI 저장을 위한 코덱
    plain_font = cv.FONT_HERSHEY_SIMPLEX
    #전체 영상
    out_recorder = cv.VideoWriter("Play_" + video_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi", fourcc, fps, (target_width, target_height))
    #결과 영상
    result_recorder = cv.VideoWriter("Result_" + video_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi", fourcc, fps, (target_width, target_height))
    result_play_recorder = cv.VideoWriter("Result_Play_" + video_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi", fourcc, fps, (target_width, target_height))
    img_select = select_img_from_video(video_file, board_pattern, out_recorder, target_width, target_height)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    # Run distortion correction
    distortion_correction(video_file, K, dist_coeff, result_play_recorder, result_recorder, target_width, target_height)

    # 모든 작업 완료 후
    out_recorder.release()
    result_recorder.release()
    cv.destroyAllWindows()
    print("Video saving completed.")
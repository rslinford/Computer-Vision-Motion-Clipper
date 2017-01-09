import os
import sys
import datetime
import imutils
import multiprocessing
import time
import datetime as dt
import cv2
import re
import numpy as np
import simplejson as json
from pathlib import Path
from dropbox.client import DropboxOAuth2FlowNoRedirect
from dropbox.client import DropboxClient
import rsl.json_config as json_config

default_config = {
   'config_file_name': r'computer-vision-motion-clipper-config.json',
   'gaussian_blur_ksize': (21, 21),
   'adaptive_threshold_ksize': 21,
   'adaptive_threshold_c': 4,
   'dilate_kernel': (7, 7),
   'dilate_iterations': 2,
   'min_contour_len': 6,
   'single_area_action_threshold': 4000,
   'post_to_dropbox_area_threshold': 40000,
   'consecutive_action_snap_threshold':  4,
   'action_detail': 8,
   'cam_source_fn_dirname': r'C:\camassets',
   'cam_ip': '192.x.x.x',
   'cam_chan_uri': r'rtsp://user:password@ip:port/Streaming/Channels/%d',
   'cam_channel_ids': [101, 201, 301, 401],
   'webcam_source_fn_dirname': r'C:\somedir\for\webcam\assets',
   'fba_source_fn_dirname': r'C:\somedir',
   'fba_source_fn_filename': r'somevideo.mp4',
   'dropbox_app_key': 'myapikey',
   'dropbox_app_secret': 'mysecret',
   'snap_action': True,
   'snap_original': False,
   'display_dropbox_uploads': False,
   'display_patch_pallet': False,
   'display_pipeline': False,
   'display_on': True,
}

def toggle_display_extra_stuff(display_extra_stuff, display_on, source_id, frame):
   cv2.destroyAllWindows()
   ensure_a_window(display_on, source_id, frame)
   return not display_extra_stuff

def toggle_display(display_on, source_id, frame):
   cv2.destroyAllWindows()
   ensure_a_window(not display_on, source_id, frame)
   return not display_on

# A window is required to receive keystrokes. If display_on is True then no worries; one will be re-created in the normal loop. 
# Otherwise, if display_on is False, then create a static window to have a window to avoid the app becoming "dead" to user input.
def ensure_a_window(display_on, source_id, frame):
   if not display_on:
      cv2.imshow("monitor_%d" % (source_id), frame)

def process_batch(config, video_source, target_fn_dirname, source_id, batch_id, dropbox_client = None):
   shrink_factor = 1 if video_source == 0 else 3

   frame_counter = 0

   # E.g. file name:
   #
   #    C:\\...\\Video Cam_3_20160613120056_20160613120350_1465851320261.mp4'
   #                         ^ the video start timestamp
   # video_start_ts is the base time for mp4 files (not streaming sources). The assumption is the mp4 file name contains a timestamp.
   video_start_ts = 0
   if video_source != 0: # video_source 0 is the local web cam. Otherwise video_sourcre is a string.
      video_start_ts_regex = re.compile(r'_%s_\d_(\d\d\d\d\d\d\d\d\d\d\d\d\d\d)_' % config['cam_ip'])
      m = video_start_ts_regex.search(video_source)
      if m:
         video_start_ts = time.mktime(time.strptime(m.groups()[0], '%Y%m%d%H%M%S'))

   # Sensitivity Tuning Knobs
   gaussian_blur_ksize = tuple(config['gaussian_blur_ksize'])
   adaptive_threshold_ksize = config['adaptive_threshold_ksize']
   adaptive_threshold_c = config['adaptive_threshold_c']
   dilate_kernel = np.ones(tuple(config['dilate_kernel']),np.uint8)
   dilate_iterations = config['dilate_iterations']
   min_contour_len = config['min_contour_len']
   single_area_action_threshold = config['single_area_action_threshold']
   post_to_dropbox_area_threshold = config['post_to_dropbox_area_threshold']
   consecutive_action_snap_threshold = config['consecutive_action_snap_threshold'] # uninterupted action above which is considered significant 
   action_detail = config['action_detail']

   # Save snapshots when significant action is dectected and snap_action is True
   snap_action = config['snap_action']
   snap_original = config['snap_original']
   display_dropbox_uploads = config['display_dropbox_uploads']
   display_patch_pallet = config['display_patch_pallet']
   display_pipeline = config['display_pipeline']

   if snap_action:
      target_fn_sub_dirname = target_fn_dirname + \
            r'\extract(%d) gbks%s atks(%d) atc(%d) dk(%d) di(%d) mcl(%d) saat(%d) ptdat(%d) cast(%d) ad(%d)' \
            % (batch_id, gaussian_blur_ksize, adaptive_threshold_ksize, adaptive_threshold_c, len(dilate_kernel), dilate_iterations, \
            min_contour_len, single_area_action_threshold, post_to_dropbox_area_threshold, consecutive_action_snap_threshold, action_detail)
   if not os.path.exists(target_fn_sub_dirname):
      try:
         os.makedirs(target_fn_sub_dirname)
      except:
         e = sys.exc_info()[0]
         print('Makedir:\n\t%s\n\tthrew error %s\nAttempting to continue.' % (target_fn_sub_dirname, str(e)))
   assert os.path.exists(target_fn_sub_dirname)
   print('Extracting images to target dir\n\t%s' % (target_fn_sub_dirname))

   snapshot_filename_template = r'bat%d_ts%f_frame%08.d_cam%02.d_%s.png'
   snapshot_full_filename_template = r'%s\%s' % (target_fn_sub_dirname, snapshot_filename_template)

   data_log_filename = r'%s\datalog_bat%d_ts%d_cam%02.d.json' % (target_fn_sub_dirname, batch_id, video_start_ts, source_id)
   print(data_log_filename)

   display_on = config['display_on']

   # Giant try block clean up resources (cameras and windows) in case of (KeyboardInterrupt, SystemExit)
   try:
      data_log  = open(data_log_filename, 'a')
      data_log_begun = False

      cap = cv2.VideoCapture(video_source)
      time.sleep(0.1)

      cap_fps = cap.get(cv2.CAP_PROP_FPS)
      previous_blur_frame = None
      consecutive_action = 0
      patches = []
      patch_pallet = None

      print('Processing %s' % (video_source))
      while True:
         frame_counter += 1

         (grabbed, original_frame) = cap.read()
         # if the frame could not be grabbed, then we have reached the end of the video
         if not grabbed:
            break

         if video_start_ts == 0:
            frame_grab_ts = time.time()
         else:
            cap_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_grab_ts = video_start_ts + (cap_pos_msec/1000)
            #print('cap_pos_msec(%s) frame_grab_ts(%s) cap_fps(%s)' % (cap_pos_msec, frame_grab_ts, cap_fps))
         #ts = dt.datetime.fromtimestamp(frame_grab_ts).strftime('%Y-%m-%d %H:%M:%S.%f')
         #print(ts)

         if patch_pallet is None:
            patch_pallet = np.zeros((original_frame.shape[0], original_frame.shape[1], 3), np.uint8)

         # resize the frame, convert it to grayscale, and blur it
         frame = imutils.resize(original_frame, width=(original_frame.shape[1] // shrink_factor))
         frame_area = frame.shape[0] * frame.shape[1]
         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         blur_frame = cv2.GaussianBlur(gray_frame, gaussian_blur_ksize, 0)

         # Deltas are calculated by diffing two blur_frames. Go back around if we don't have a previous_blur_frame yet.
         if previous_blur_frame is None:
            previous_blur_frame = blur_frame
            previous_frame_grab_ts = frame_grab_ts
            if not display_on:
               # Show a window with a static image. Display is off; however, a window is required to accept keystrokes. 
               cv2.imshow("monitor_%d" % (source_id), frame)
            continue

         elapsed_since_previous = frame_grab_ts - previous_frame_grab_ts
         previous_frame_grab_ts = frame_grab_ts

         # Calculate delta 
         base_delta_frame = cv2.absdiff(previous_blur_frame, blur_frame)
         previous_blur_frame = blur_frame

         # Extract contours
         base_threshold_frame = cv2.adaptiveThreshold(base_delta_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_threshold_ksize, adaptive_threshold_c)
         dilate_threshold_frame = cv2.dilate(base_threshold_frame, dilate_kernel, iterations=dilate_iterations)
         (_, contours, _) = cv2.findContours(dilate_threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         # Analyze contours
         action_detected = False
         total_area = 0
         trail = []
         big_enough_contours = []
         for c in contours:
            if len(c) < min_contour_len:
               continue

            # transform ellipse data into a drawable
            ellipse = cv2.fitEllipse(c)
            (cx, cy) = ellipse[0]
            (ax1, ax2) = ellipse[1]
            angle = int(ellipse[2])
            center = (int(cx), int(cy))
            axes = (int(ax1) ,int(ax2))
            trail.append([frame_grab_ts, center, axes, angle])

            area = cv2.contourArea(c)
            total_area += area
            big_enough_contours.append(c)

         gr_color = 255
         for e in trail:
            cv2.ellipse(frame, e[1], e[2], e[3], 0 , 360, (255, gr_color, gr_color), 1)
            gr_color -= 15

         br_color = 0
         patches_log_item = []
         if (big_enough_contours):
            cv2.drawContours(frame, big_enough_contours, -1, (br_color, 255, br_color), 1)
            patch_index = 0
            for c in big_enough_contours:
               (x, y, w, h) = cv2.boundingRect(c)

               # Draw red rectangles fadding to white
               cv2.rectangle(frame, (x, y), (x+w, y+h), (br_color, br_color, 255), 1)
               br_color += 15

               # Extract the patch bound by the rectangle
               x = x * shrink_factor
               y = y * shrink_factor
               w = w * shrink_factor
               h = h * shrink_factor
               x2 = x + w
               y2 = y + h
               patch = original_frame[y:y2, x:x2, :]

               # Write big patches to disk. Small patches are slapped on the patch_pallet as noise.
               area = cv2.contourArea(c)
               if (area > single_area_action_threshold):
                  action_detected = True
                  # Hold big patches in memory to be saved to disk later if consecutive threshold is met.
                  patch_index += 1
                  tag = r'x%dy%dw%dh%d_%d_patch' % (x,y,w,h, patch_index)
                  fn = snapshot_full_filename_template % (batch_id, frame_grab_ts, frame_counter, source_id, tag)
                  patches_log_item.append(os.path.split(fn)[1])
                  patches.append((fn, patch, (x,y), (x2,y2), area))
               else:
                  # Small patches are useful as noise on the patch_pallet. Draw them first so big ones will overwrite noise.
                  if frame_counter % 10 == 0:
                     patch_pallet[y:y2, x:x2] = patch

         consecutive_action = consecutive_action + 1 if action_detected else 0
         if not consecutive_action and patches:
            patches = []

         # draw the text and timestamp on the frame
         text_color = (0, 0, 255) if action_detected else (255, 255, 100)
         motion_event_triggered = ((consecutive_action == consecutive_action_snap_threshold) or (consecutive_action > consecutive_action_snap_threshold and consecutive_action % action_detail == 0))
         text_thickness = 2 if motion_event_triggered else 1

         instantaneous_frames_per_second = 1 / elapsed_since_previous

         len_contour = len(big_enough_contours)
         average_contour_size = (total_area / len_contour) if len_contour > 0 else 0

         status_text = 'Event(%d)(%03.d)(%d) Area(%08.d) Cont(%04.d) Ave(%07.1d) E(%03.2f)' % \
            (action_detected, consecutive_action, motion_event_triggered, total_area, len_contour, average_contour_size, instantaneous_frames_per_second)
         cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, text_thickness)
         cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

         data_point = {'fgts':frame_grab_ts, 'fc':frame_counter, 'sid':source_id, 'ad':action_detected, 'ca':consecutive_action, 'met':motion_event_triggered, \
            'ta':total_area, 'lc':len_contour, 'acs':average_contour_size, 'ifps':instantaneous_frames_per_second, 'fn':None, 'fn_patches':patches_log_item}

         if motion_event_triggered:

            if snap_action:
               fn = snapshot_full_filename_template % (batch_id, frame_grab_ts, frame_counter, source_id, 'marked')
               data_point['fn'] = os.path.split(fn)[1]
               print('Writing image ' + fn)
               cv2.imwrite(fn, frame)
               if total_area > post_to_dropbox_area_threshold:
                  if display_dropbox_uploads:
                     cv2.imshow('Posted cam(%s) frame(%d)' % (source_id, frame_counter), frame)
                  if dropbox_client is not None:
                     print('\tDropbox upload cam(%s) size(%s)' % (source_id, total_area))
                     dropbox_client.put_file('/%s' % (os.path.split(fn)[1]), open(fn, 'rb'))

               if snap_original:
                  fn_original = snapshot_full_filename_template % (batch_id, frame_grab_ts, frame_counter, source_id, 'original')
                  cv2.imwrite (fn_original, original_frame)
               i = 0
               for (patch_fn, patch, p1, p2, area) in patches:
                  # Write patches to disk
                  print('\t%s' % (os.path.split(patch_fn)[1]))
                  cv2.imwrite(patch_fn, patch)
                  # Draw some to the patch_pallet. Only a fraction are drawn to maintain patchiness.
                  if i % 4 == 0:
                     patch_pallet[p1[1]:p2[1], p1[0]:p2[0]] = patch
                  i += 1
               patches = []
            else:
               print('<Shutter Sound> frame %d' % (frame_counter))

         if data_log_begun:
            data_log.write(',')
         else:
            data_log.write('[')
            data_log_begun = True
         data_log.write('%s\n' % (json.dumps(data_point)))

         if display_on:
            cv2.imshow('monitor_%d' % (source_id), frame)
            if display_pipeline:
               cv2.imshow('base_delta_%d' % (source_id), base_delta_frame)
               cv2.imshow('base_threshold_%d' % (source_id), base_threshold_frame)
               cv2.imshow('dilate_threshold_%d' % (source_id), dilate_threshold_frame)
            if display_patch_pallet:
               cv2.imshow('Pallet %d' % (source_id), patch_pallet)
 
         key = cv2.waitKey(1) & 0xFF
         if key == ord('q'):
            raise KeyboardInterrupt
         elif key == ord('p'):
            display_pipeline = toggle_display_extra_stuff(display_pipeline, display_on, source_id, frame)
         elif key == ord('P'):
            display_patch_pallet = toggle_display_extra_stuff(display_patch_pallet, display_on, source_id, frame)
         elif key == ord('s'):
            display_on = toggle_display(display_on, source_id, frame)
   except (KeyboardInterrupt):
      print('Program ended per user reqeust.')
   except (SystemExit):
      print('Program terminated. Attempting to clean resources.')
   finally:
      print('Releasing camera source: %s' % (video_source))
      cap.release()
      print('Destroying any OpenCV')
      cv2.destroyAllWindows()
      print('Closing log file %s' % data_log_filename)
      data_log.write(']')
      data_log.close()

def mp_worker(inputs):
    start_time = int(time.time())
    stamp = 'Processs (%s, %s, %s, %s)' % (inputs[1], inputs[2], inputs[3], inputs[4])
    print ('Begin %s' % (stamp))
    process_batch(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
    elapsed_time = time.time() - start_time
    print ('End(%d) %s' %(elapsed_time, stamp))
    
def mp_handler(inputs_array):
    p = multiprocessing.Pool(10)
    p.map(mp_worker, inputs_array)

def connect_to_dropbox(config):
   # connect to dropbox and start the session authorization process
   app_key = config['dropbox_app_key']
   app_secret = config['dropbox_app_secret']
   auth_flow = DropboxOAuth2FlowNoRedirect(app_key, app_secret)
   print ('[INFO] Authorize this application: %s' % (auth_flow.start()))
   authCode = input('Enter auth code here: ').strip()
 
   # finish the authorization and grab the Dropbox client
   (accessToken, userID) = auth_flow.finish(authCode)
   dropbox_client = DropboxClient(accessToken)
   print ('[SUCCESS] dropbox account linked')
   return drop_client

def main():
   config_file_name = default_config['config_file_name']
   # Command line config takes precedence if one is found
   if len(sys.argv) > 1:
      if os.path.isfile(sys.argv[1]):
         cmd_line_filename = os.path.normpath(sys.argv[1])
         full_path_file_stem, ext = os.path.splitext(cmd_line_filename)
         ext = ext.lower()
         if ext == '.json':
            config_file_name = cmd_line_filename
   try:
      config = json_config.load(config_file_name)
      json_config.normalize(config, default_config)
   except (FileNotFoundError):
      json_config.create_default(default_config)
      config = json_config.load(default_config['config_file_name'])

   upload_them_sweet_action_pics = False
   dropbox_client = connect_to_dropbox(config) if upload_them_sweet_action_pics else None

   batch_id = int(time.time())

   # Process single mp4 video file. Single thread.
   if False:
     source_fn_dirname = config['fba_source_fn_dirname']
     process_batch(config, r'%s\%s' % (source_fn_dirname, config['fba_source_fn_filename']), source_fn_dirname, 0, batch_id, dropbox_client)
     return

  # Local Web Cam single thread.
   if True:
      process_batch(config, 0, config['webcam_source_fn_dirname'], 0, batch_id, dropbox_client)
      return

   # Process all mp4 files in a directory. Multi-threaded.
   if False:
     source_fn_dirname = config['fyh_source_fn_dirname']
     paths = [x for x in Path(source_fn_dirname).iterdir() if x.suffix == '.mp4']
     batch_source = ([config, str(paths[i]), source_fn_dirname, i, batch_id, dropbox_client] for i in range(len(paths)))
     mp_handler(batch_source)
     return

  # CCTV cam feeds multi-threaded.
   if False:
     source_fn_dirname = config['cam_source_fn_dirname'] # Cameras are the source. Target dir is created below the "source" dir.
     chan_uri = config['cam_chan_uri']
     channel_ids = config['cam_channel_ids']
     batch_source = ([config, chan_uri % (channel_ids[i]), source_fn_dirname, i, batch_id, dropbox_client] for i in range(len(channel_ids)))
     if False:
        # Call a single channel synchronously for debugging
        process_batch(config, chan_uri % (channel_ids[2]), source_fn_dirname, 2, batch_id, dropbox_client)
        return
     mp_handler(batch_source)

if __name__ == '__main__':
     main()

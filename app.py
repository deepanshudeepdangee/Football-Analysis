import streamlit as st
import os
import pickle
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import cv2

# Create folders for inputs, outputs, and cache
INPUT_FOLDER = 'input_videos'
OUTPUT_FOLDER = 'output_videos'
CACHE_FOLDER = 'cache_data'   # Folder to save cached results
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

def save_cache(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_cache(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Inject custom CSS for styling
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}
div.stFileUploader > label {
    font-weight: 600;
    color: #2a9d8f;
    font-size: 18px;
}
h1 {
    font-weight: 800 !important;
    color: #264653;
}
.stButton > button {
    background-color: #2a9d8f;
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 10px 24px;
    transition: background-color 0.3s ease;
    box-shadow: 0 4px 8px rgba(42, 157, 143, 0.3);
}
.stButton > button:hover {
    background-color: #21867a;
}
.stDownloadButton > button {
    background-color: #e76f51 !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 8px rgba(231, 111, 81, 0.3);
}
.stDownloadButton > button:hover {
    background-color: #d65a3a !important;
}
[data-testid="stSidebar"] h3 {
    color: #e63946;
    font-weight: 700;
}
video {
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.markdown("""
### How to Use
- Upload one or more football match videos (formats: mp4, avi, mov, mkv).
- Click **Process Video** and wait for the analysis to complete.
- Preview and download the processed video.
- ⚠️ Large videos take longer to process.

Made with ❤️ by  Nityam """)

st.title("⚽ Football Video Analysis")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to disk
    input_video_path = os.path.join(INPUT_FOLDER, uploaded_file.name)
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded video saved to `{input_video_path}`")

    # Get video info (frames, resolution)
    cap = cv2.VideoCapture(input_video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    st.markdown(f"**Video info:** {num_frames} frames, Resolution: {width}x{height}")

    cache_path = os.path.join(CACHE_FOLDER, uploaded_file.name.rsplit('.',1)[0] + '_cache.pkl')
    output_filename = uploaded_file.name.rsplit('.', 1)[0] + '_output.avi'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    if st.button("▶️ Process Video"):
        if os.path.exists(cache_path) and os.path.exists(output_path):
            with st.spinner("🔄 Loading cached data and output video..."):
                cached_data = load_cache(cache_path)
                tracks = cached_data['tracks']
                camera_movement_per_frame = cached_data['camera_movement_per_frame']
                video_frames = cached_data['video_frames']
                team_ball_control = cached_data.get('team_ball_control', [])
            st.success("✅ Loaded cached processing data!")

            # Show cached processed video and details
            col1, col2 = st.columns([2,1])
            with col1:
                st.video(output_path)
            with col2:
                st.markdown("### Video Details")
                st.write(f"**Frames:** {len(video_frames)}")
                st.write(f"**Resolution:** {width}x{height}")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name=output_filename,
                        mime="video/avi",
                    )
        else:
            with st.spinner("🎥 Processing video, this may take some time..."):
                video_frames = read_video(input_video_path)

                tracker = Tracker('models/best.pt')
                tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
                tracker.add_position_to_tracks(tracks)

                camera_movement_estimator = CameraMovementEstimator(video_frames[0])
                camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=False)
                camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

                view_transformer = ViewTransformer()
                view_transformer.add_transformed_position_to_tracks(tracks)

                tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

                speed_and_distance_estimator = SpeedAndDistance_Estimator()
                speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

                team_assigner = TeamAssigner()
                team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

                for frame_num, player_track in enumerate(tracks['players']):
                    for player_id, track in player_track.items():
                        team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                        tracks['players'][frame_num][player_id]['team'] = team
                        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

                player_assigner = PlayerBallAssigner()
                team_ball_control = []
                for frame_num, player_track in enumerate(tracks['players']):
                    ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', None)
                    if ball_bbox is None:
                        team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
                        continue
                    assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
                    if assigned_player != -1:
                        tracks['players'][frame_num][assigned_player]['has_ball'] = True
                        team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                    else:
                        team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

                cached_data = {
                    'tracks': tracks,
                    'camera_movement_per_frame': camera_movement_per_frame,
                    'video_frames': video_frames,
                    'team_ball_control': team_ball_control,
                }
                save_cache(cache_path, cached_data)
                st.success("✅ Video processed and cached!")

            # Draw annotations and save output video
            tracker = Tracker('models/best.pt')  # Reinit tracker for drawing
            output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

            min_len = min(len(output_video_frames), len(camera_movement_per_frame))
            output_video_frames = output_video_frames[:min_len]
            camera_movement_per_frame = camera_movement_per_frame[:min_len]

            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

            speed_and_distance_estimator = SpeedAndDistance_Estimator()
            speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

            save_video(output_video_frames, output_path)

            # Display video and download button in two columns
            col1, col2 = st.columns([2,1])
            with col1:
                st.video(output_path)
            with col2:
                st.markdown("### Video Details")
                st.write(f"**Frames:** {len(output_video_frames)}")
                st.write(f"**Resolution:** {width}x{height}")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name=output_filename,
                        mime="video/avi",
                    )

else:
    st.info("Please upload a video file to start.")

import os
import cv2
from pathlib import Path

# آدرس پوشه ویدیوها
VIDEOS_DIR = r"E:\Amirhossein\youtube_videos"  # این را تغییر بده

# فرمت‌های قابل پشتیبانی
VIDEO_EXTENSIONS = {".mpg", ".mpeg", ".mp4"}

def extract_frames(video_path, interval_seconds=4):
    video_name = video_path.stem

    # پوشه خروجی با نام خود ویدیو
    output_dir = video_path.parent / video_name
    output_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ خطا در باز کردن: {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"❌ FPS نامعتبر برای: {video_path.name}")
        cap.release()
        return

    frame_interval = int(fps * interval_seconds)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0
    saved_count = 0

    print(f"🎬 در حال پردازش: {video_path.name}")

    while True:
        success, frame = cap.read()

        if not success:
            break

        if current_frame % frame_interval == 0:
            frame_filename = (
                f"{video_name}_frame_{current_frame:08d}.jpg"
            )

            frame_path = output_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)

            saved_count += 1

        current_frame += 1

    cap.release()

    print(
        f"✅ {video_path.name} | "
        f"کل فریم‌ها: {total_frames:,} | "
        f"فریم ذخیره‌شده: {saved_count:,}"
    )


def main():
    videos_dir = Path(VIDEOS_DIR)

    if not videos_dir.exists():
        print("❌ پوشه پیدا نشد")
        return

    video_files = [
        f for f in videos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print("❌ هیچ ویدیویی پیدا نشد")
        return

    print(f"📁 تعداد ویدیوها: {len(video_files)}")

    for video_file in video_files:
        extract_frames(video_file, interval_seconds=1)

    print("🎉 همه ویدیوها پردازش شدند.")


if __name__ == "__main__":
    main()
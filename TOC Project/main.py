from src.yolo_detection import detect_objects
from src.real_time_detection import process_video
from src.database_integration import insert_video_details
from src.image_database_integration import insert_image_details
from src.alert_message import main as send_alert


def main():
    print("=== Disaster Alert System ===")
    print("1. Detect objects in images and update database")
    print("2. Process videos for real-time detection and update database")
    print("3. Send alert email with GPS coordinates")
    print("4. Compare YOLOv5 and YOLOv8 models")
    print("5. Exit")

    choice = input("Choose an option (img/vid): ").lower()

    images = [
        "C:\\Users\\Akshay Prakash\\Documents\\Semester 4\\TOC\\TOC Project\\src\\image7.png",
        "C:\\Users\\Akshay Prakash\\Documents\\Semester 4\\TOC\\TOC Project\\src\\image8.png"
    ]

    videos = [
        "C:\\Users\\Akshay Prakash\\Documents\\Semester 4\\TOC\\TOC Project\\src\\video1.mp4",
        "C:\\Users\\Akshay Prakash\\Documents\\Semester 4\\TOC\\TOC Project\\src\\video2.mp4"
    ]

    if choice == "img":
        for image_path in images:
            image_name = image_path.split("\\")[-1]
            print(f"\n🔍 Processing Image: {image_name}")
            detected_classes = detect_objects(image_path)
            insert_image_details(image_name, image_path)

            if detected_classes:
                print("🚨 Alert: Person or animal detected in the image!")
                send_alert()
            else:
                print("✅ No person or animal detected in the image.")

    elif choice == "vid":
        for video_path in videos:
            video_name = video_path.split("\\")[-1]
            print(f"\n🎥 Processing Video: {video_name}")
            detected_classes = process_video(video_path)
            insert_video_details(video_name, video_path)

            if detected_classes:
                print("🚨 Alert: Person or animal detected in the video!")
                send_alert()
            else:
                print("✅ No person or animal detected in the video.")

    else:
        print("⚠️ Invalid option. Please choose 'img' or 'vid'.")


if __name__ == "__main__":
    main()

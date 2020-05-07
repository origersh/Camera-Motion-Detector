"""
# 
# MotionDetect: record moving object on screen using my camera
# 
"""
import cv2, time, pandas
from datetime import datetime

class MotionDetect:
    def __init__(self):
        self.first_frame = None          # prepare to capture first frame in the video
        self.status_list = [None, None]  # save status states - will be converted to graph
        self.times = []                  # save time signatures of changes

        # create data frame to store motion changes and export them to a csv file
        self.data_frame = pandas.DataFrame(columns=["Start", "End"])

        # assign camera to record video
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    ### Track moving object in the video captured by the camera
    def track_motion(self):
        while True:
            try:
                # capture a single frame
                return_value, frame = self.camera.read()
                # save the status of when changes are diaplayed
                status=0
                # convert the frame to gray-scale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # smooth the image
                gray = cv2.GaussianBlur(gray, (15,15), 0)
                
                # capture first frame in gray-scale
                if self.first_frame is None:
                    self.first_frame = gray
                    continue
                
                # getting the difference between the first_frame and current frame
                delta_frame = cv2.absdiff(self.first_frame, gray)
                # assigning a (binary) threshold to the image to convert every pixel value that is less that 30 to 255
                thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
                # amplifying the difference
                thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
                # find the contours of the movement displayed by the white pixels
                ## better to use a copy of the original image
                contours, hierarchy = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # filter the contours to display only changed pixels
                for contour in contours:
                    # if contour have an erae of less than 1000 pixels - continue
                    if cv2.contourArea(contour) < 10000:
                        continue
                    # change state to 1:"capturing changes"
                    status=1
                    # draw rect around contour
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                # add status to status list
                self.status_list.append(status)
                
                # write the time when changes appear
                if self.status_list[-1] == 1 and self.status_list[-2] == 0:
                    self.times.append(datetime.now())
                if self.status_list[-1] == 0 and self.status_list[-2] == 1:
                    self.times.append(datetime.now())
                
                # Displaying the frame
                # cv2.imshow("gray_frame", gray)
                # cv2.imshow("delta_frame", delta_frame)
                # cv2.imshow("threshed_frame", thresh_frame)
                cv2.imshow("motion detect", frame)

                # break on 'q' press
                if cv2.waitKey(100) == ord('q'):
                    if status == 1: self.times.append(datetime.now())
                    break
            except Exception as e:
                raise e
                break

        # exit the program
        self.camera.release()
        cv2.destroyAllWindows()


    ### Export time stamps of changes to csv file
    def export_to_csv(self):
        # building the data frame based on time signatures of the changes
        for i in range(0, len(self.times), 2):
            self.data_frame = self.data_frame.append({"Start":self.times[i], "End":self.times[i+1]}, ignore_index=True)

        # export data to new file
        self.data_frame.to_csv("Movement.csv")


if __name__ == "__main__":
    md = MotionDetect()
    md.track_motion()
    md.export_to_csv()
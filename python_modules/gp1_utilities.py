import json
import socket
import cv2 as cv


def create_json_info(center_list):
    dictionary = {"number_of_packs" : len(center_list),
                  "centers_list" : center_list
                  }
    json_str = json.dumps(dictionary, indent=4, default=str)
    return json_str

def send_via_udp(message, UDP_IP = "10.2.86.116", UDP_PORT = 5060):
    MESSAGE = message
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))

def send_results(center_list, UDP_IP = "10.2.86.116", UDP_PORT = 5060):
    message = create_json_info(center_list)
    send_via_udp(message, UDP_IP, UDP_PORT)

def select_frame(cap):
    while (True):
        ret, frame = cap.read()
        if ret == True:
            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('b'):
                back = frame
                break

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv.destroyAllWindows()
    return back

if __name__ == "__main__":
    center_list = [[373.0, 122.0], [352.0, 192.5], [607.0, 203.5], [720.75, 205.5], [809.5, 411.75]]
    send_results(center_list)
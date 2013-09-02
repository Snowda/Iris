#!/usr/bin/env python

import numpy as np
import cv2, getopt, sys, re, datetime, urllib2, twitter

api = twitter.Api(consumer_key='',
            consumer_secret='', access_token_key='',
            access_token_secret='')
user = ""

def create_capture(source = 0, 
    fallback = 'synth:class=chess:bg=../cpp/lena.jpg:noise=0.1:size=640x480'):
    '''source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    '''
    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv2.VideoCapture(source)
        if 'size' in params:
            w, h = map(int, params['size'].split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print 'Warning: unable to open video source: ', source
        if fallback is not None:
            return create_capture(fallback, None)
    return cap

def clock():
    """"""
    return cv2.getTickCount() / cv2.getTickFrequency()

def draw_str(dst, (x, y), s, front_color=(255,255,255)):
    #cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, front_color, lineType=cv2.LINE_AA)

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def detect(img, cascade, old_rects=None):
    if old_rects == None:
        old_rects = []
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return old_rects
    rects[:,2:] += rects[:,:2]

    for x1, y1, x2, y2 in rects:
        x_size = x2-x1
        y_size = y2-y1
        rect_pyth = (y_size*y_size)+(x_size*x_size)

    for x1, y1, x2, y2 in old_rects:
        x_size = x2-x1
        y_size = y2-y1
        old_pyth = (y_size*y_size)+(x_size*x_size)

        if old_pyth > rect_pyth:
            rects = ((old_rects)*2 + (rects)) / 3

    return rects

def face_mask(image, mask, shape):
    for x1, y1, x2, y2 in shape:
        x_size = x2-x1
        y_size = y2-y1
        
        if type(shape) != list:
            scaled_mask = cv2.resize(mask, (x_size, y_size))
        else:
            scaled_mask = mask

        #image[y1:y1+scaled_mask.shape[0], x1:x1+scaled_mask.shape[1]] = scaled_mask

        for c in range(0,3):
            image[y1:y1+scaled_mask.shape[0], x1:x1+scaled_mask.shape[1], c] = scaled_mask[:,:,c] * (scaled_mask[:,:,3]/255.0) + image[y1:y1+scaled_mask.shape[0], x1:x1+scaled_mask.shape[1], c] * (1.0 - scaled_mask[:,:,3]/255.0)

def display_fps(image, this_time):
    """"""
    frame_time = 1/(clock() - this_time)
    draw_back_str(image, (20, 20), 'FPS: %.2f' % (frame_time))

    return clock()

def text_hover(image, face, text_data):
    for x1, y1, x2, y2 in face:
        x_center = x1+x2/2
        width = image.shape[1]
        if width/2 < x_center:
            x_axis = 20
        else:
            x_axis = x2
        for position, key in enumerate(text_data):
            if y1 >= 30:
                draw_back_str(image, (x_axis, (y1-20+(20*position))), key, rect_color=text_data[key][1], text_color=text_data[key][0])
            else:
                draw_back_str(image, (x_axis, 30+(20*position)), key, rect_color=text_data[key][1], text_color=text_data[key][0])

def draw_eyes(image, gray, rects, nested, old_rects):
    for x1, y1, x2, y2 in rects:
        roi = gray[y1:y2, x1:x2]
        vis_roi = image[y1:y2, x1:x2]
        subrects = detect(roi.copy(), nested, old_rects)
        draw_rects(vis_roi, subrects, (255, 0, 0))

def internet_on():
    """Checks if the internet connetion is working"""
    return target_online('http://www.google.com') #google is always online

def target_online(url_to_check, return_string=False):
    """Checks if the supplied URL is online"""
    #headers = {'User-Agent' : 'Mozilla 23.0'} # Add your headers
    try:
        connection = urllib2.urlopen(url_to_check, timeout=2) # Create the Request.
    except urllib2.URLError, e:
        if(return_string):
            if internet_on():
                print "Connetion to target timed out. Try again."
            else :
                print "No internet connection. Check your connectivity."
            print "Error: "+str(e)
        return False
    else:
        if(return_string):
            return connection
        else:
            return True

def get_country_name():
    """
    Using IP lookup tables, checks user's location. Won't work behind proxy
    """
    ip_check_url = 'http://checkip.dyndns.org'
    ip_data = target_online(ip_check_url, return_string=True)
    if ip_data:
        response = re.search(
            re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"), ip_data).group()
        geo_ip = pygeoip.GeoIP('data/GeoIP.dat')
        return geo_ip.country_name_by_name(response) #country_code_by_name
    else :
        return "Location not found"

def suffix(day):
    """function which picks a date suffix depending on the date"""
    if(11 <= day <= 13) :
        return 'th' 
    else:
        return { 1 : "st", 2 : "nd", 3 : "rd" }.get(day % 10, "th")

def todays_date():
    """returns today's date properly formatted"""
    right_now = datetime.datetime.now()
    return right_now.strftime('%B {S}, %Y').replace('{S}', 
        str(right_now.day) + suffix(right_now.day))

def print_over_old(print_string):
    """Prints a new line to the terminal and removes the previous line"""
    sys.stdout.write("\r")
    sys.stdout.write("                                                        ") 
    sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.write(print_string) 
    sys.stdout.flush()

def generate_data():
    data_dict = {}

    data_dict["Conor Forde"] = [(255,0,0), (0,0,0)]
    data_dict["Age | 22"] = [(0,0,255), (0,0,0)]

    #country = get_country_name()
    #data_list.append(str(country))

    hour = datetime.datetime.now().hour
    if hour <= 7:
        data_dict["Should be asleep"] = [(0,255,0), (0,0,0)]
    else:
        data_dict["Should be working"] = [(0,255,0), (0,0,0)]

    if 0xFF & cv2.waitKey(1) == ord('w'):
        data_dict["pressed w"] = [(255,255,255), (0,0,0)]
    elif 0xFF & cv2.waitKey(1) == ord('a'):
        data_dict["pressed a"] = [(255,255,255), (0,0,0)]
    elif 0xFF & cv2.waitKey(1) == ord('s'):
        data_dict["pressed s"] = [(255,255,255), (0,0,0)]
    elif 0xFF & cv2.waitKey(1) == ord('d'):
        data_dict["pressed d"] = [(255,255,255), (0,0,0)]
    else:
        data_dict["pressed nothing"] = [(255,255,255), (0,0,0)]

    return data_dict

def timetable():
    """"""
    data_dict = {}
    data_dict[" 9:00am Email Visa Office"] = [(255,255,255), (0,0,0)]
    data_dict["10:00am Talk to recruiters"] = [(255,255,255), (0,0,0)]
    data_dict["12:00am Update Git Repos"] = [(255,255,255), (0,0,0)]
    data_dict[" 2:00pm Get shampoo"] = [(255,255,255), (0,0,0)]
    data_dict[" 6:00pm Go to IoT Meetup"] = [(255,255,255), (0,0,0)]

    return data_dict

def todo_list():
    """"""
    data_dict = {}
    data_dict["1 | Email Visa Office"] = [(255,255,255), (0,0,0)]
    data_dict["2 | Talk to recruiters"] = [(255,255,255), (0,0,0)]
    data_dict["3 | Update Git Repos"] = [(255,255,255), (0,0,0)]
    data_dict["4 | Get shampoo"] = [(255,255,255), (0,0,0)]
    data_dict["5 | Go to IoT Meetup"] = [(255,255,255), (0,0,0)]

    return data_dict
    
def twitter():
    """"""
    data_list = []
    statuses = api.GetUserTimeline(user)

    for s in statuses[:5]:
        for x in range(0, 7):
            charso = 20*x
            
            data_list.append(s.text[charso-20:charso])
        data_list.append(" ")

    #data_list.append("Conor Forde")
    #data_list.append("@MyOuterWorld")
    #data_list.append("Tweets (726)")
    #data_list.append("Following (164)")
    #data_list.append("Followers (50)")
    return data_list

    data_dict = {}
    data_dict[""] = [(255,255,255), (0,0,0)]
    data_dict[""] = [(255,255,255), (0,0,0)]
    data_dict[""] = [(255,255,255), (0,0,0)]
    data_dict[""] = [(255,255,255), (0,0,0)]
    data_dict[""] = [(255,255,255), (0,0,0)]

    return data_dict
    
def settings():
    """"""
    data_dict = {}
    data_dict["1. Keyboard Settings"] = [(255,255,255), (0,0,0)]
    data_dict["2. Video Input"] = [(255,255,255), (0,0,0)]
    data_dict["3. Audio"] = [(255,255,255), (0,0,0)]
    data_dict["4. Privacy Settings"] = [(255,255,255), (0,0,0)]
    data_dict["5. User Settings"] = [(255,255,255), (0,0,0)]

    return data_dict
    
def music():
    """"""
    data_dict = {}
    data_dict["Firestarter"] = [(255,255,255), (0,0,0)]
    data_dict["The Prodigy"] = [(255,255,255), (0,0,0)]
    data_dict["1:45 / 2:33"] = [(255,255,255), (0,0,0)]
    data_dict["(N)ext / (P)revious"] = [(255,255,255), (0,0,0)]
    data_dict["(S)top"] = [(255,255,255), (0,0,0)]

    return data_dict
    
def facebook():
    """"""
    data_dict = {}
    data_dict["Conor Forde"] = [(255,255,255), (0,0,0)]
    data_dict["233 Friends"] = [(255,255,255), (0,0,0)]
    data_dict["(0) Friend Requests"] = [(255,255,255), (0,0,0)]
    data_dict["(1) New Message"] = [(255,255,255), (0,0,0)]
    data_dict["(2) Updates"] = [(255,255,255), (0,0,0)]

    return data_dict
    
def time():
    """"""
    weekday = datetime.datetime.now().strftime("%A")

    data_dict = {}
    data_dict[todays_date()] = [(255,255,255), (0,0,0)]
    data_dict["San Francisco | CA"] = [(255,255,255), (0,0,0)]

    return data_dict
    
def shopping_list():
    """"""
    data_dict = {}
    data_dict["1. Shampoo"] = [(255,255,255), (0,0,0)]
    data_dict["2. Toilet Paper"] = [(255,255,255), (0,0,0)]
    data_dict["3. Eirdinger"] = [(255,255,255), (0,0,0)]
    data_dict["4. Phealeh Ticket"] = [(255,255,255), (0,0,0)]

    return data_dict
    
def skills():
    """"""
    data_dict = {}
    data_dict["Skills"] = [(255,255,255), (0,0,0)]
    data_dict["C | C++ | Python"] = [(255,255,255), (0,0,0)]
    data_dict["OpenCV"] = [(255,255,255), (0,0,0)]
    data_dict["Linux | Git"] = [(255,255,255), (0,0,0)]
    data_dict["Embedded Systems"] = [(255,255,255), (0,0,0)]

    return data_dict

def business_card():
    """"""
    data_dict = {}
    data_dict["Conor Forde"] = [(255,255,255), (0,0,0)]
    data_dict["Electronic Design Engineer"] = [(255,255,255), (0,0,0)]
    data_dict["me@conorforde.com"] = [(255,255,255), (0,0,0)]
    data_dict["(415) 423 4026"] = [(255,255,255), (0,0,0)]
    data_dict["linkedin.com/in/conorforde"] = [(255,255,255), (0,0,0)]

    return data_dict

def corner_display(image):
    hour = str(datetime.datetime.now().hour)
    minute = str(datetime.datetime.now().minute)

    #y_size = image.shape[0]
    x_size = image.shape[1]

    battery_data = open("/sys/class/power_supply/BAT1/capacity", "r").read()
    battery_percent = re.sub(r"\?", "", str(battery_data))
    draw_back_str(image, ((x_size - 60), 20), hour+":"+minute)
    draw_back_str(image, ((x_size - 220), 20), "Battery: "+battery_percent+"%")

def draw_back_str(image, x_by_y, text, text_color=(255,255,255), rect_color=(0,0,0), alpha=0.8, padding=5):
    before = image.copy()

    height = 19

    other_x = x_by_y[0]
    other_y = x_by_y[1]
    text_len = len(text)
    rect_len = text_len*10

    cv2.rectangle(image, (other_x-padding, other_y+padding), (other_x+rect_len-padding, other_y-height+padding), rect_color, -1)

    neg_alpha = 1 - alpha
    cv2.addWeighted(before, alpha, image, neg_alpha, 0, image)
    draw_str(image, x_by_y, text, text_color)

#incomming call
def detect_hands(img):
    """"""
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, (cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU))
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
        cnt=contours[ci]
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    cv2.drawContours(drawing,[hull],0,(0,0,255),2)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull) 
    mind=0
    maxd=0
    i=0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        dist = cv2.pointPolygonTest(cnt,centr,True)
        cv2.line(img,start,end,[0,255,0],2)                
        cv2.circle(img,far,5,[0,0,255],-1)
        print(i)

def read_keyboard(data_list, option_dict):
        if (0xFF & cv2.waitKey(1) == ord('q')) and (current != option_list["q"]):
            data_list = business_card()
            current = option_list["q"]
        elif (0xFF & cv2.waitKey(1) == ord('w')) and (current != option_list["w"]):
            data_list = time()
            current = option_list["w"]
        elif (0xFF & cv2.waitKey(1) == ord('e')) and (current != option_list["e"]):
            data_list = facebook()
            current = option_list["e"]
        elif (0xFF & cv2.waitKey(1) == ord('r'))and (current != option_list["r"]):
            data_list = timetable()
            current = option_list["r"]
        elif (0xFF & cv2.waitKey(1) == ord('t')) and (current != option_list["t"]):
            data_list = todo_list()
            current = option_list["t"]
        elif (0xFF & cv2.waitKey(1) == ord('y')) and (current != option_list["y"]):
            data_list = music()
            current = option_list["y"]
        elif (0xFF & cv2.waitKey(1) == ord('u')) and (current != option_list["u"]):
            data_list = settings()
            current = option_list["u"]
        elif (0xFF & cv2.waitKey(1) == ord('i')) and (current != option_list["i"]):
            data_list = shopping_list()
            current = option_list["i"]
        elif (0xFF & cv2.waitKey(1) == ord('o')) and (current != option_list["o"]):
            #data_list = twitter()
            current = option_list["o"]
        elif (0xFF & cv2.waitKey(1) == ord('p')) and (current != option_list["p"]):
            data_list = skills()
            current = option_list["p"]
        else:
            current = "default"

        print_over_old(current)
        return data_list

def main():
    cascade = cv2.CascadeClassifier(
        "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested = cv2.CascadeClassifier(
        "../../data/haarcascades/haarcascade_eye.xml")
    hogg = cv2.CascadeClassifier(
        "../../data/hogcascade/hogcascades_pedestrians.xml")

    cam = create_capture(0)
    old_rects = []
    x_offset=y_offset=0

    data_list = generate_data()
    time_delay = clock()

    option_list = { "q": "business card", "w": "time", "e": "facebook", 
        "r": "timetable", "t": "todo list", "y": "music", "u": "settings", 
        "i": "shopping list", "o": "twitter", "p": "skills"}

    for option in option_list:
        print option+" : "+option_list[option]

    while True:
        data_list = read_keyboard(data_list, option_list)
        ret, img = cam.read()
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        #face = detect_hands(img)

        rects = detect(gray, cascade, old_rects)
        old_rects = rects

        vis = img.copy()

        #width, height = cv2.frameSize(vis)
        #draw_rects(vis, rects, (0, 255, 0))

        text_hover(vis, rects, data_list)
        #time_delay = display_fps(vis, time_delay)

        corner_display(vis)
        cv2.imshow('Project Iris', vis)

        if 0xFF & cv2.waitKey(1) == 27:
            break
    print "\nExiting"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

"""

Display analog data from Nuvoton using Python (matplotlib)
"""

import sys, serial, argparse
import numpy as np
from time import sleep, perf_counter
from collections import deque

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import os

# plot class
class AnalogPlot:
  # constr
  def __init__(self, strPort, maxLen, output_file_name, show_print):
      # open serial port
      self.ser = serial.Serial(strPort,115200)
      #self.ser = serial.Serial(strPort,115200, bytesize=8, parity='N', stopbits=1, timeout=1)

      self.ax = deque([0.0]*maxLen)
      self.ay = deque([0.0]*maxLen)
      self.az = deque([0.0]*maxLen)
      self.maxLen = maxLen
      
      self.axyz = np.zeros((3, maxLen))
      self.output_file_name = output_file_name
      self.show_print = show_print

  # add to buffer
  def addToBuf(self, buf, val):
      if len(buf) < self.maxLen:
          buf.append(val)
      else:
          #buf.pop()
          #buf.appendleft(val)
          buf.popleft()
          buf.append(val)

  # add data
  def add(self, data):
      #assert(len(data) == 3)
      self.addToBuf(self.ax, data[0])
      self.addToBuf(self.ay, data[1])
      self.addToBuf(self.az, data[2])
     

  # update plot
  def update(self, frameNum, a0, a1,a2):
      try:
          raw_line = self.ser.readline()
          try:
              line = raw_line.decode()
              #print(line)
              data = [float(val) for val in line.split(',')]
              
              # print data
              if(len(data) == 3):
                  print("{},{},{}".format(data[0],data[1],data[2]))
                  self.add(data)
                  a0.set_data(range(self.maxLen), self.ax)
                  a1.set_data(range(self.maxLen), self.ay)
                  a2.set_data(range(self.maxLen), self.az)

          except:
            pass  #decode fail
          
      except KeyboardInterrupt:
          print('exiting')
      
      return a0,
  
  def save_text2file(self):
      
      # Check the already exist data number
      if os.path.exists(self.output_file_name):
           def blocks(files, size=1024):
               while True:
                   b = files.read(size)
                   if not b: break
                   yield b
           with open(self.output_file_name, "r",encoding="utf-8",errors='ignore') as f:
               data_number = sum(bl.count("\n") for bl in blocks(f))
               data_number = (data_number // (self.maxLen + 2)) + 1 # 1 space & 1 "-,-,-,num"
      else: # first new data
          data_number = 1        
              
      print("How many data so far in {}: {}".format(self.output_file_name, data_number))
      
      # Writing all data to a file 
      with open(self.output_file_name, "a") as file1:
          file1.write("\n")
          file1.write("-,-,-,{}\n".format(data_number))
          for idx , val in enumerate(self.axyz[0]):
              file1.write("{},{},{} \n".format(self.axyz[0][idx], self.axyz[1][idx], self.axyz[2][idx]))
              #file1.writelines(L)
    
  def add_text(self, data, count):
      self.axyz[0][count-1] = data[0]
      self.axyz[1][count-1] = data[1]
      self.axyz[2][count-1] = data[2]
  
  def text_only(self):
      try:
          count = 0
          while count < self.maxLen:
              while self.ser.in_waiting:          # 若收到序列資料…
                  try:
                      raw_line = self.ser.readline()  # 讀取一行
                      line = raw_line.decode()   # 用預設的UTF-8解碼
                      data = [float(val) for val in line.split(',')]
                      if(len(data) == 3):
                          count += 1
                          self.add_text(data, count)
                          if self.show_print:
                              print("{},{},{}".format(data[0],data[1],data[2]))
                  except:
                      pass  #decode fail 
           
          self.save_text2file()
          self.plot_plt()            
          print("Collect finish, total 3X{} points.".format(self.maxLen))
          
      except KeyboardInterrupt:
          self.ser.close()    # 清除序列通訊物件
          print('再見！') 

  def plot_plt(self):
      fig = plt.figure("Show analog data (www.nuvoton.com.tw)")
      ax = plt.axes(xlim=(0, self.maxLen), ylim=(-3000,3000))
      a0, = ax.plot( self.axyz[0], label='X')
      a1, = ax.plot( self.axyz[1], label='Y')
      a2, = ax.plot( self.axyz[2], label='Z')
      ax.legend(loc='upper right')
      plt.xlabel('Time (1/100 s)')
      plt.ylabel('Acc')
      plt.title('M460 Accelerometer (www.nuvoton.com.tw)')
      plt.show()

  # clean up
  def close(self):
      # close serial
      self.ser.flush()
      self.ser.close()    

def create_tag_folder(tag_name):
    dir_path = os.path.join(os.getcwd(), tag_name)
    try:
        os.mkdir(dir_path)
    except OSError as error:
        print(error)
        print('skip create')

# main() function
def main():
  # create parser
  parser = argparse.ArgumentParser(description="LDR serial")
  # add expected arguments
  parser.add_argument('--port', dest='port', required=True)
  parser.add_argument(
        '--user_name',
        type=str,
        default='cy',
        help='The action name, ex: data/{action_label}/output_{action_label}_cy')
  parser.add_argument(
        '--action_label',
        type=str,
        default='wing',
        help='The action name, ex: data/wing/output_wing_{user_name}')
  parser.add_argument(
        '--max_len',
        type=int,
        default=200,
        help='The length of collecting points each time. Output speed is 100HZ, so the time of each collecting is about max_len*(1/100) seconds')
  parser.add_argument(
        '--negative_num',
        type=str,
        default=1,
        help='ex: output_negative_1.txt, output_negative_2.txt, ...')
  parser.add_argument(
        '--show_print',
        type=int,
        default=0,
        help='1: show the print value. 0: not show')

  # parse args
  args = parser.parse_args()
  
  # prepare the raw data folder & file
  strPort = args.port
    
  if 'negative' in args.action_label.lower():
      output_dir_path = os.path.join("data", "negative")
      create_tag_folder(output_dir_path)
      output_file_path = os.path.join(output_dir_path, (r"output_negative" + r"_" + args.negative_num + r".txt"))
  else:
      output_dir_path = os.path.join("data", args.action_label.lower())
      create_tag_folder(output_dir_path)    
      output_file_path = os.path.join(output_dir_path, (r"output_" + args.action_label.lower() + r"_" + args.user_name + r".txt")) 
  print("The path of output file: {}".format(output_file_path))

  print('reading from serial port %s...' % strPort)
  print("Preapare to move target board, and countdown from 2!!!")
  for r in range(2): 
      print("{} !!".format((2 - r)))
      sleep(1)
  print("Start!!!")
  sleep(0.5) # for human reaction time 
  
  # Start to collect data from seriel port, save to file & plot picture.
  analogPlot = AnalogPlot(strPort, args.max_len, output_file_path, args.show_print)
  analogPlot.text_only()

  ## set up animation
  #fig = plt.figure("Show analog data (www.nuvoton.com.tw)")
  #ax = plt.axes(xlim=(0, 1000), ylim=(-3000,3000))
  #a0, = ax.plot([], [],label='X')
  #a1, = ax.plot([], [],label='Y')
  #a2, = ax.plot([], [],label='Z')
  #ax.legend(loc='upper right')
  #plt.xlabel('Time (s)')
  #plt.ylabel('Acc')
  #plt.title('M460 Accelerometer (www.nuvoton.com.tw)')
  #
  #anim = animation.FuncAnimation(fig, analogPlot.update, 
  #                               fargs=(a0, a1,a2), 
  #                               interval=1)
#
  ## show plot
  #plt.show()
  
  # clean up
  analogPlot.close()

  print('exiting.')
  

# call main
if __name__ == '__main__':
  main()
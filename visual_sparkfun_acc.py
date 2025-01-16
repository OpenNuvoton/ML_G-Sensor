"""
Display analog data from Nuvoton using Python (matplotlib)
"""
import os
import argparse
from time import sleep
from collections import deque
import serial

import numpy as np
import matplotlib.pyplot as plt


# plot class
class AnalogPlot:
    """
    A class to handle the plotting and saving of accelerometer data from a serial port.
    """
    def __init__(self, str_port, max_len, output_file_name, show_print):
        # open serial port
        self.ser = serial.Serial(str_port, 115200)
        #self.ser = serial.Serial(str_port,115200, bytesize=8, parity='N', stopbits=1, timeout=1)

        self.ax = deque([0.0]*max_len)
        self.ay = deque([0.0]*max_len)
        self.az = deque([0.0]*max_len)
        self.max_len = max_len

        self.axyz = np.zeros((3, max_len))
        self.output_file_name = output_file_name
        self.show_print = show_print

    # add to buffer
    def add2buf(self, buf, val):
        """
        Add a value to the buffer. If the buffer is full, remove the oldest value.
        Args:
            buf (collections.deque): The buffer to which the value will be added.
            val (any): The value to add to the buffer.
        Returns:
            None
        """
        if len(buf) < self.max_len:
            buf.append(val)
        else:
            #buf.pop()
            #buf.appendleft(val)
            buf.popleft()
            buf.append(val)

    # add data
    def add(self, data):
        """
        Adds the provided data to the respective buffers for ax, ay, and az.
        """
        #assert(len(data) == 3)
        self.add2buf(self.ax, data[0])
        self.add2buf(self.ay, data[1])
        self.add2buf(self.az, data[2])

    # update plot
    def update(self, a0, a1,a2):
        """
        Update the plot with new data from the serial port.
        This function reads a line of data from the serial port, decodes it, and 
        updates the plot with the new accelerometer data if the data is valid.
        Returns:
        tuple: A tuple containing the updated line object for the x-axis data.
        """
        try:
            raw_line = self.ser.readline()
            try:
                line = raw_line.decode()
                data = [float(val) for val in line.split(',')]
                # print data
                if len(data) == 3:
                    print(f"{data[0]},{data[1]},{data[2]}")
                    self.add(data)
                    a0.set_data(range(self.max_len), self.ax)
                    a1.set_data(range(self.max_len), self.ay)
                    a2.set_data(range(self.max_len), self.az)
            except ValueError:
                pass  # decode fail
        except KeyboardInterrupt:
            print('exiting')

        return a0

    def save_text2file(self):
        """
        Saves the accelerometer data to a file.
        This method checks if the output file already exists. If it does, it calculates the number of existing data entries
        and appends the new data with an incremented data number. If the file does not exist, it starts with the first data entry.
        The data is written in the following format:
        - A newline character
        - A line with the format "-,-,-,data_number"
        - Multiple lines with the format "x,y,z"
        The data is sourced from the `axyz` attribute of the class instance.
        """

        # Check the already exist data number
        if os.path.exists(self.output_file_name):
            def blocks(files, size=1024):
                while True:
                    b = files.read(size)
                    if not b:
                        break
                    yield b
            with open(self.output_file_name, "r",encoding="utf-8",errors='ignore') as f:
                data_number = sum(bl.count("\n") for bl in blocks(f))
                data_number = (data_number // (self.max_len + 2)) + 1 # 1 space & 1 "-,-,-,num"
        else: # first new data
            data_number = 1

        print(f"How many data so far in {self.output_file_name}: {data_number}")

        # Writing all data to a file
        with open(self.output_file_name, "a", encoding="utf-8") as file1:
            file1.write("\n")
            file1.write(f"-,-,-,{data_number}\n")
            for idx , _ in enumerate(self.axyz[0]):
                file1.write(f"{self.axyz[0][idx]},{self.axyz[1][idx]},{self.axyz[2][idx]} \n")

    def add_text(self, data, count):
        """
        Updates the axyz array with new data at the specified count index.
        """
        self.axyz[0][count-1] = data[0]
        self.axyz[1][count-1] = data[1]
        self.axyz[2][count-1] = data[2]

    def text_only(self):
        """
        Reads data from a serial port, processes it, and saves it to a file.
        This method continuously reads lines from the serial port until a specified 
        number of lines (self.max_len) have been read. Each line is expected to contain 
        three comma-separated float values. The method decodes the line, splits it into 
        individual float values, and processes the data. If the data is valid, it is 
        added to a text buffer and optionally printed to the console.
        After collecting the required number of lines, the method saves the collected 
        data to a file and generates a plot. If a KeyboardInterrupt is detected, the 
        serial port is closed and a goodbye message is printed.
        """
        try:
            count = 0
            while count < self.max_len:
                while self.ser.in_waiting:          # if collect the serial data
                    try:
                        raw_line = self.ser.readline()  # read one line
                        line = raw_line.decode()   # UTF-8
                        data = [float(val) for val in line.split(',')]
                        if len(data) == 3:
                            count += 1
                            self.add_text(data, count)
                            if self.show_print:
                                print(f"{data[0]},{data[0]},{data[0]}")
                    except ValueError:
                        pass  # decode fail

            self.save_text2file()
            self.plot_plt()
            print(f"Collect finish, total 3X{self.max_len} points.")

        except KeyboardInterrupt:
            self.ser.close()    # clear the serial port
            print('ByeBye！')

    def plot_plt(self):
        """
        Plots the accelerometer data using matplotlib.
        This function creates a plot with three lines representing the X, Y, and Z
        axes of the accelerometer data. The plot displays the data over time with
        the X-axis labeled as 'Time (1/100 s)' and the Y-axis labeled as 'Acc'.
        """
        _ = plt.figure("Show analog data (www.nuvoton.com.tw)")
        ax = plt.axes(xlim=(0, self.max_len), ylim=(-3000,3000))
        _, = ax.plot( self.axyz[0], label='X')
        _, = ax.plot( self.axyz[1], label='Y')
        _, = ax.plot( self.axyz[2], label='Z')
        ax.legend(loc='upper right')
        plt.xlabel('Time (1/100 s)')
        plt.ylabel('Acc')
        plt.title('M460 Accelerometer (www.nuvoton.com.tw)')
        plt.show()

    # clean up
    def close(self):
        """
        Closes the serial connection.
        """
        # close serial
        self.ser.flush()
        self.ser.close()

def create_tag_folder(tag_name):
    """
    Creates a folder with the specified tag name in the current working directory.
    """
    dir_path = os.path.join(os.getcwd(), tag_name)
    try:
        os.mkdir(dir_path)
    except OSError as error:
        print(error)
        print('skip create')

def main():
    """
    Main function to parse command-line arguments, prepare output directories and files,
    and collect data from a serial port.
    """
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
    str_port = args.port

    if 'negative' in args.action_label.lower():
        output_dir_path = os.path.join("data", "negative")
        create_tag_folder(output_dir_path)
        output_file_path = os.path.join(output_dir_path, (r"output_negative" + r"_" + args.negative_num + r".txt"))
    else:
        output_dir_path = os.path.join("data", args.action_label.lower())
        create_tag_folder(output_dir_path)
        output_file_path = os.path.join(output_dir_path, (r"output_" + args.action_label.lower() + r"_" + args.user_name + r".txt"))
    print(f"The path of output file: {output_file_path}")

    print(f'reading from serial port {str_port}...')
    print("Preapare to move target board, and countdown from 2!!!")
    for r in range(2):
        print(f"{(2 - r)} !!")
        sleep(1)
    print("Start!!!")
    sleep(0.5) # for human reaction time

    # Start to collect data from seriel port, save to file & plot picture.
    analog_plot = AnalogPlot(str_port, args.max_len, output_file_path, args.show_print)
    analog_plot.text_only()

    # clean up
    analog_plot.close()
    print('exiting.')

# call main
if __name__ == '__main__':
    main()

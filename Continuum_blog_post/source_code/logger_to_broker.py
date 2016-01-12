#!/usr/bin/env python
from __future__ import division

import pika
import time
import argparse
import urllib2
import math

retry_timer = 30

# Coefficients for ref voltage Type K TC range 0 to 1372C 
T_ref = 25
b0 = -1.7600413686*(10**-2)
b1 = 3.8921204975*(10**-2)
b2 = 1.8558770032*(10**-5)
b3 = -9.9457592874*(10**-8)
b4 = 3.1840945719*(10**-10)
b5 = -5.6072844889*(10**-13)
b6 = 5.6075059059*(10**-16)
b7 = -3.2020720003*(10**-19)
b8 = 9.7151147152*(10**-23)
b9 = -1.2104721275*(10**-26)

# Coefficients for Temp Type K TC 0 to 500 C
c0 = 0
c1 = 2.508355 * 10**1
c2 = 7.860106 * 10**-2
c3 = -2.503131 * 10**-1
c4 = 8.315270 * 10**-2
c5 = -1.228034 * 10**-2
c6 = 9.804036 * 10**-4
c7 = -4.413030 * 10**-5
c8 = 1.057734 * 10**-6
c9 = -1.052755 * 10**-8 

# Coefficients for SBG
m = 0.308857142857
b = 0.00142857142857

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('broker', help='Network or IP address of message broker')
parser.add_argument('logger_id', help='Logger name or ID')
parser.add_argument('log_file', help='Location of log file')
args = parser.parse_args()

def read_voltage(channel):
    # Read voltage from specified ADC channel over REST API
    response = urllib2.urlopen('http://localhost/arduino/adc/' + str(channel))
    voltage = response.read()
    response.close()
    voltage = float(voltage)
    return voltage

def calculate_T(V):
    v_ref = (b0 + b1*T_ref + b2*T_ref**2 + b3*T_ref**3 + b4*T_ref**4 + 
    	b5*T_ref**5 + b6*T_ref**6 + b7*T_ref**7 + b8*T_ref**8 + b9*T_ref**9)
    V = V + float(v_ref)
    T = c0+ c1*V + c2*V**2 + c3*V**3 + c4*V**4 + c5*V**5 + c6*V**6 + c7*V**7 + c8*V**8 + c9*V**9
    T = round(float(T), 1)
    return T

def calculate_HF(voltage):
    HF = (voltage-zero_voltage)*m + b
    HF = round(float(HF), 2)
    return HF

# Attemps to connect to server and run data broadcast loop.
# If it fails to connect to the broker, it will wait some time
# and attempt to reconnect indefinitely.
while True:
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=args.broker))
        channel = connection.channel()
        channel.exchange_declare(exchange='logs', type='fanout')

        # Read voltage data, convert to T and HF, send T and HF
        while True:
            # Read voltages from ADC channels
            T_voltage = read_voltage(1)
            HF_voltage = read_voltage(2)

            # Calculate temperature, HF
            T = calculate_T(T_voltage)
            HF = calculate_HF(HF_voltage)

            # Construct message and publish to broker
            message = (time.ctime()+',%s,%d,%0.1f,%0.2f') % (args.logger_id, total_time, T, HF)
            channel.basic_publish(exchange='logs', routing_key='', body=message)
            print 'Sent %r' % (message)

            # Save data to microSD cars
            with open(args.log_file, 'a+') as text_file:
                text_file.write(message+'\n')
            
            total_time = total_time + 1
            time.sleep(1)

        connection.close()
    except:
        print 'No broker found. Retrying in 30 seconds...'
        timer = 0
        while timer < retry_timer:
            # Read voltages from ADC channel
            T_voltage = read_voltage(1)
            HF_voltage = read_voltage(2)

            # Calculate temperature, HF
            T = calculate_T(T_voltage)
            HF = calculate_HF(HF_voltage)

            # Construct message for log
            message = (time.ctime()+',%s,%d,%0.1f,%0.2f') % (args.logger_id, total_time, T, HF)
            with open(args.log_file, 'a+') as text_file:
                text_file.write(message+'\n')
            total_time = total_time + 1
            time.sleep(1)
            timer += 1

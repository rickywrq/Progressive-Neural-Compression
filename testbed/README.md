# Testbed

## Radio Module and Setup
### nRF52840 development kits
The device and the server communicate with each other the via a pair of [nRF52840 development kits](https://www.nordicsemi.com/Products/Development-hardware/nrf52840-dk) manufactured by Nordic Semiconductor. The kit offers a serial port interface that allows the client and the server to transmit and receive data streams with the IEEE 802.15.4 radio at the application level. The MAC layer of the radio uses re-transmissions to overcome transmission failures.

The **nRF5 SDK** is required to configure the nRF52840 development kits. [[Download]](https://www.nordicsemi.com/Products/Development-software/nRF5-SDK#Downloads) [[Documentation]](https://infocenter.nordicsemi.com/topic/sdk_nrf5_v17.1.0/getting_started_installing.html)

The source code and the project file of the interface (Wireless UART Example) can be found in https://infocenter.nordicsemi.com/topic/sdk_nrf5_v17.1.0/wireless_uart_15_4.html
* Note that we used a higher Baud rate of 230400 to achieve a higher transmission rate. 

### Connecting the DK to the device and the edge server.
You may connect the nRF52840 DK using the USB-A to Micro USB cable. The DK will show up in the system, e.g., `/dev/ttyACM0`.

Now, the device and the server can communicate using the pySerial library.
```
serial.Serial('/dev/ttyACM0', 230400, rtscts=True)  # open a serial port
```
#include <iostream>
#include <chrono>
#include <thread>

// Simulated frequency readings for Red, Green, and Blue light
int redFrequency = 100;
int greenFrequency = 150;
int blueFrequency = 200;

// Simulated function to get frequency (This would come from hardware in Arduino)
int getFrequency(char color) {
    switch(color) {
        case 'R': return redFrequency;
        case 'G': return greenFrequency;
        case 'B': return blueFrequency;
        default: return 0;
    }
}

// Convert frequency to wavelength (in nanometers)
float frequencyToWavelength(int frequency) {
    if (frequency > 150 && frequency <= 300) {
        return 450;  // Blue light wavelength (450nm)
    } else if (frequency > 100 && frequency <= 150) {
        return 550;  // Green light wavelength (550nm)
    } else if (frequency <= 100) {
        return 650;  // Red light wavelength (650nm)
    }
    return -1;  // Invalid frequency
}

int main() {
    while (true) {
        // Simulate reading frequencies
        int redFreq = getFrequency('R');
        int greenFreq = getFrequency('G');
        int blueFreq = getFrequency('B');

        // Convert to wavelength
        float redWavelength = frequencyToWavelength(redFreq);
        float greenWavelength = frequencyToWavelength(greenFreq);
        float blueWavelength = frequencyToWavelength(blueFreq);

        // Output the wavelength
        std::cout << "Red Light Wavelength: " << redWavelength << " nm" << std::endl;
        std::cout << "Green Light Wavelength: " << greenWavelength << " nm" << std::endl;
        std::cout << "Blue Light Wavelength: " << blueWavelength << " nm" << std::endl;

        // Delay for readability
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}

// Définition des broches de sortie
const int outputPin1 = 13; // Bispectrale
const int outputPin3 = 9; // Ximea

// Définition des variables pour les fréquences des signaux créneaux
const unsigned long period1 = 50000; //BISPECTRALE -  Période en microsecondes (1 / 20 Hz = 50 ms = 50000 µs)
const unsigned long period3 = 10000; //XIMEA -  Période en microsecondes (1 / 100 Hz = 10 ms = 10000 µs)

// Seuils pour la détection des fronts montants et descendants
const float thresholdHigh = 3.0; // Seuil pour détecter le front montant (en volts)
const float thresholdLow = 1.0;  // Seuil pour détecter le front descendant (en volts)

// Variables pour stocker l'état des sorties et l'état précédent du signal
volatile bool outputState1 = false; // État de la sortie 1 (LOW au départ)
volatile bool outputState3 = false; // État de la sortie 3 (LOW au départ)

float previousInputVoltage = 0.0;  // Valeur précédente du signal d'entrée

float previousMicros1 = 0;
float previousMicros3 = 0;

void setup() {
  pinMode(outputPin1, OUTPUT);
  pinMode(outputPin3, OUTPUT);

  Serial.begin(115200); // Initialisation du port série à 9600 bauds
}

void loop() {
  // Lecture de la tension d'entrée (signal créneau)
  float inputVoltage = analogRead(A0) * (5.0 / 1023.0); // Lecture de A0 et conversion en volts (0-5V)

  // Détection des fronts montants et descendants
  if (inputVoltage > thresholdHigh && previousInputVoltage <= thresholdHigh) {
    // Front montant détecté
    outputState1 = true;
    outputState3 = true;

    //Serial.println("Front montant détecté");
  } else if (inputVoltage < thresholdLow && previousInputVoltage >= thresholdLow) {
    // Front descendant détecté
    outputState1 = false;
    outputState3 = false;
    
    //Serial.println("Front descendant détecté");
  }

  // Mise à jour de l'état des sorties
  updateOutputs();
  Serial.println(inputVoltage);

  // Enregistrement de la valeur actuelle du signal d'entrée pour la prochaine itération
  previousInputVoltage = inputVoltage;

}

// Fonction pour mettre à jour l'état des sorties en fonction de outputState1 et outputState2
void updateOutputs() {
  // Génération du signal créneau pour la bispectrale
  if (outputState1) {
    unsigned long currentMicros = micros();
    if (currentMicros - previousMicros1 >= period1 / 2) {
      previousMicros1 = currentMicros;
      digitalWrite(outputPin1, !digitalRead(outputPin1)); // Inverser l'état de la sortie 1
    }
  } else {
    digitalWrite(outputPin1, LOW); // Mettre la sortie 1 à LOW
  }

  // Génération du signal créneau pour les Ximea
  if (outputState3) {
    unsigned long currentMicros = micros();
    if (currentMicros - previousMicros3 >= period3 / 2) {
      previousMicros3 = currentMicros;
      digitalWrite(outputPin3, !digitalRead(outputPin3)); // Inverser l'état de la sortie 3
    }
  } else {
    digitalWrite(outputPin3, LOW); // Mettre la sortie 3 à LOW
  }

 
}

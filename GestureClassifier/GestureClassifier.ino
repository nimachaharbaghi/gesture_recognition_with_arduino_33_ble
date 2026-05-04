// gesture_model.h  — add this tiny header to your sketch folder
// ---------------------------------------------------------------
// #pragma once
// extern const unsigned char gesture_model_data[];
// extern const unsigned int  gesture_model_data_len;
// ---------------------------------------------------------------

// GestureClassifier.ino
// Reads IMU, collects a stroke, extracts features,
// runs TFLite inference, then lights Blue/Red/Green LED.

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "gesture_model.h"

// ── Tunable parameters (must match Python preprocessing) ──────────
#define RESAMPLE_LEN   64
#define NUM_FEATURES  134    // 64*2 + 6 stats
#define NUM_CLASSES     3

#define CAPTURE_THRESH  2.5f  // g – start recording when accel > this
#define MAX_SAMPLES      200  // raw IMU samples per gesture
#define CONFIDENCE_MIN  0.70f // ignore predictions below this

// ── TFLite arena ──────────────────────────────────────────────────
constexpr int kArenaSize = 10 * 1024;
uint8_t tensor_arena[kArenaSize];

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

// ── Raw capture buffer ─────────────────────────────────────────────
float raw_x[MAX_SAMPLES], raw_y[MAX_SAMPLES];
int   n_samples = 0;

// ── LED helpers ───────────────────────────────────────────────────
// Arduino Nano 33 BLE: built-in RGB is ACTIVE LOW
void ledOff()  { digitalWrite(LEDR,HIGH); digitalWrite(LEDG,HIGH); digitalWrite(LEDB,HIGH); }
void ledBlue() { ledOff(); digitalWrite(LEDB, LOW); }   // w
void ledRed()  { ledOff(); digitalWrite(LEDR, LOW); }   // o
void ledGreen(){ ledOff(); digitalWrite(LEDG, LOW); }   // l

// ── Resample helper ───────────────────────────────────────────────
// Linearly interpolate `src` (len n) into `dst` (len RESAMPLE_LEN)
void linearResample(const float* src, int n, float* dst) {
    for (int i = 0; i < RESAMPLE_LEN; i++) {
        float t   = (float)i / (RESAMPLE_LEN - 1) * (n - 1);
        int   lo  = (int)t;
        int   hi  = min(lo + 1, n - 1);
        float frac = t - lo;
        dst[i] = src[lo] + frac * (src[hi] - src[lo]);
    }
}

// ── Feature extraction (mirrors Python stroke_to_features) ────────
void buildFeatureVector(float* feat) {
    float rx[RESAMPLE_LEN], ry[RESAMPLE_LEN];
    linearResample(raw_x, n_samples, rx);
    linearResample(raw_y, n_samples, ry);

    // Bounding-box normalisation to [-1, 1]
    float vmin =  1e9f, vmax = -1e9f;
    for (int i = 0; i < RESAMPLE_LEN; i++) {
        if (rx[i] < vmin) vmin = rx[i];  if (rx[i] > vmax) vmax = rx[i];
        if (ry[i] < vmin) vmin = ry[i];  if (ry[i] > vmax) vmax = ry[i];
    }
    float rng = (vmax - vmin > 1e-6f) ? (vmax - vmin) : 1.0f;
    for (int i = 0; i < RESAMPLE_LEN; i++) {
        rx[i] = (rx[i] - vmin) / rng * 2.0f - 1.0f;
        ry[i] = (ry[i] - vmin) / rng * 2.0f - 1.0f;
    }

    // Interleave x, y
    for (int i = 0; i < RESAMPLE_LEN; i++) {
        feat[2*i]   = rx[i];
        feat[2*i+1] = ry[i];
    }

    // 6 summary statistics
    float mx=0,sx=0,my=0,sy=0,mdx=0,mdy=0;
    for (int i = 0; i < RESAMPLE_LEN; i++) { mx += rx[i]; my += ry[i]; }
    mx /= RESAMPLE_LEN;  my /= RESAMPLE_LEN;
    for (int i = 0; i < RESAMPLE_LEN; i++) {
        sx  += (rx[i]-mx)*(rx[i]-mx);
        sy  += (ry[i]-my)*(ry[i]-my);
    }
    sx = sqrt(sx / RESAMPLE_LEN);  sy = sqrt(sy / RESAMPLE_LEN);
    for (int i = 0; i < RESAMPLE_LEN-1; i++) {
        mdx += rx[i+1]-rx[i];
        mdy += ry[i+1]-ry[i];
    }
    mdx /= (RESAMPLE_LEN-1);  mdy /= (RESAMPLE_LEN-1);

    int base = RESAMPLE_LEN * 2;
    feat[base+0]=mx; feat[base+1]=sx;
    feat[base+2]=my; feat[base+3]=sy;
    feat[base+4]=mdx; feat[base+5]=mdy;
}

// ── Setup ─────────────────────────────────────────────────────────
void setup() {
    Serial.begin(9600);
    pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
    ledOff();

    if (!IMU.begin()) {
        Serial.println("IMU init failed!");
        while (1);
    }

    // Load TFLite model
    tfl_model = tflite::GetModel(gesture_model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        tfl_model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();

    input  = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Ready — move the board to draw a gesture (w / o / l)");
}

// ── Loop ──────────────────────────────────────────────────────────
void loop() {
    float ax, ay, az;

    // Wait for significant movement to start capture
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        float mag = sqrt(ax*ax + ay*ay + az*az);

        if (mag > CAPTURE_THRESH) {
            n_samples = 0;
            Serial.println("Capturing...");

            // Collect until motion stops or buffer full
            while (n_samples < MAX_SAMPLES) {
                if (IMU.accelerationAvailable()) {
                    IMU.readAcceleration(ax, ay, az);
                    raw_x[n_samples] = ax;
                    raw_y[n_samples] = ay;
                    n_samples++;
                    float m2 = sqrt(ax*ax + ay*ay + az*az);
                    // Stop if board is still for 20 consecutive samples
                    static int still_count = 0;
                    if (m2 < 1.2f) still_count++; else still_count = 0;
                    if (still_count > 20) break;
                }
            }

            if (n_samples < 10) { Serial.println("Too short, ignored."); return; }

            // Build feature vector
            float features[NUM_FEATURES];
            buildFeatureVector(features);

            // Copy to model input tensor
            for (int i = 0; i < NUM_FEATURES; i++)
                input->data.f[i] = features[i];

            // Run inference
            if (interpreter->Invoke() != kTfLiteOk) {
                Serial.println("Inference failed!");
                return;
            }

            // Read probabilities
            float p_w = output->data.f[0];
            float p_o = output->data.f[1];
            float p_l = output->data.f[2];

            Serial.print("w="); Serial.print(p_w, 2);
            Serial.print(" o="); Serial.print(p_o, 2);
            Serial.print(" l="); Serial.println(p_l, 2);

            // Pick winner
            int pred = 0;
            float best = p_w;
            if (p_o > best) { best = p_o; pred = 1; }
            if (p_l > best) { best = p_l; pred = 2; }

            if (best < CONFIDENCE_MIN) {
                Serial.println("Low confidence — ignoring.");
                ledOff();
                return;
            }

            // Light the correct LED
            if      (pred == 0) { ledBlue();  Serial.println("→ w  (BLUE)");  }
            else if (pred == 1) { ledRed();   Serial.println("→ o  (RED)");   }
            else                { ledGreen(); Serial.println("→ l  (GREEN)"); }

            delay(2000);  // keep LED on for 2 s
            ledOff();
        }
    }
}
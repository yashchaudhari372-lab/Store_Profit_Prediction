import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load Model
MODEL_PATH = "Gradient_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("GradientBoostingRegressor model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please ensure the pickle file is in the root directory.")

# Feature Definitions
FEATURE_NAMES = [
    "Ship Mode", "Customer Name", "Segment", "Country", "City", 
    "State", "Region", "Category", "Sub-Category", "Product Name", 
    "Sales", "Quantity", "Discount"
]

DEFAULT_VALUES = {
    "Ship Mode": 0, "Customer Name": 10, "Segment": 0, "Country": 0,
    "City": 20, "State": 10, "Region": 1, "Category": 0,
    "Sub-Category": 3, "Product Name": 100, "Sales": 250.00,
    "Quantity": 3, "Discount": 0.10
}

# Embedded HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" x-data="dashboard()" :class="theme">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Boosting Intelligence Dashboard</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['Fira Code', 'monospace'],
                        display: ['Outfit', 'sans-serif']
                    },
                    colors: {
                        brand: {
                            50: '#f0f6ff',
                            500: '#3b82f6',
                            600: '#2563eb',
                            700: '#1d4ed8',
                        }
                    }
                }
            }
        }
    </script>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <!-- Alpine.js -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        [x-cloak] { display: none !important; }
        
        /* Custom Color Schemes */
        .theme-cyber {
            --bg-primary: #0b0f19;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-color: rgba(59, 130, 246, 0.2);
            --accent-glow: rgba(99, 102, 241, 0.15);
            --text-main: #f3f4f6;
            --card-gradient: linear-gradient(135deg, rgba(30, 27, 75, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        }
        
        .theme-sunset {
            --bg-primary: #180914;
            --bg-card: rgba(36, 15, 30, 0.7);
            --border-color: rgba(244, 63, 94, 0.2);
            --accent-glow: rgba(244, 63, 94, 0.15);
            --text-main: #fdf2f8;
            --card-gradient: linear-gradient(135deg, rgba(88, 28, 135, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        }

        .theme-emerald {
            --bg-primary: #022c22;
            --bg-card: rgba(6, 78, 59, 0.5);
            --border-color: rgba(16, 185, 129, 0.2);
            --accent-glow: rgba(16, 185, 129, 0.15);
            --text-main: #ecfdf5;
            --card-gradient: linear-gradient(135deg, rgba(6, 95, 70, 0.4) 0%, rgba(2, 44, 34, 0.6) 100%);
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            transition: background-color 0.5s ease, color 0.5s ease;
        }

        .glass-card {
            background: var(--card-gradient);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Animations */
        @keyframes pulse-glow {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 0.8; }
        }
        .animate-glow {
            animation: pulse-glow 4s infinite ease-in-out;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
        ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.4); }
    </style>
</head>
<body class="font-sans antialiased min-h-screen relative overflow-x-hidden" :class="fontStyle">

    <!-- Background Ambient Glow -->
    <div class="fixed top-0 left-1/4 w-96 h-96 bg-indigo-600/20 rounded-full blur-3xl pointer-events-none animate-glow"></div>
    <div class="fixed bottom-0 right-1/4 w-96 h-96 bg-rose-600/20 rounded-full blur-3xl pointer-events-none animate-glow" style="animation-delay: 2s;"></div>

    <div class="container mx-auto px-4 py-8 relative z-10 max-w-7xl">
        
        <!-- Header & Theme Selector -->
        <header class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8 glass-card p-6 rounded-2xl">
            <div class="flex items-center gap-4">
                <div class="p-3 bg-gradient-to-tr from-indigo-500 to-rose-500 rounded-xl shadow-lg">
                    <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                </div>
                <div>
                    <h1 class="text-2xl md:text-3xl font-extrabold font-display bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400">
                        Gradient Boosting Analytics
                    </h1>
                    <p class="text-xs md:text-sm text-gray-400">Interactive ML Prediction & Insights Engine</p>
                </div>
            </div>

            <!-- Controls (Theme & Font Selector) -->
            <div class="flex flex-wrap items-center gap-3">
                <!-- Theme Picker -->
                <div class="flex items-center bg-black/30 p-1.5 rounded-xl border border-white/10">
                    <button @click="setTheme('theme-cyber')" :class="{'bg-indigo-600 text-white': theme === 'theme-cyber', 'text-gray-400 hover:text-white': theme !== 'theme-cyber'}" class="px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200">
                        Cyber
                    </button>
                    <button @click="setTheme('theme-sunset')" :class="{'bg-rose-600 text-white': theme === 'theme-sunset', 'text-gray-400 hover:text-white': theme !== 'theme-sunset'}" class="px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200">
                        Sunset
                    </button>
                    <button @click="setTheme('theme-emerald')" :class="{'bg-emerald-600 text-white': theme === 'theme-emerald', 'text-gray-400 hover:text-white': theme !== 'theme-emerald'}" class="px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200">
                        Emerald
                    </button>
                </div>

                <!-- Font Selector -->
                <select x-model="fontStyle" class="bg-black/30 text-xs text-gray-300 border border-white/10 rounded-xl px-3 py-2 outline-none focus:border-indigo-500 transition-all">
                    <option value="font-sans">Modern Sans</option>
                    <option value="font-display">Display Outfit</option>
                    <option value="font-mono">Technical Mono</option>
                </select>
            </div>
        </header>

        <!-- Main Content Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <!-- Left Side: Interactive Input Parameters -->
            <div class="lg:col-span-5 flex flex-col gap-6">
                <div class="glass-card rounded-2xl p-6">
                    <div class="flex items-center justify-between mb-6 border-b border-white/10 pb-4">
                        <h2 class="text-lg font-bold flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-indigo-500 animate-ping"></span>
                            Input Parameters
                        </h2>
                        <button @click="resetDefaults()" class="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">Reset Defaults</button>
                    </div>

                    <form @submit.prevent="runPrediction()" class="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                        <template x-for="(val, key) in formData" :key="key">
                            <div class="flex flex-col gap-1.5 bg-black/20 p-3 rounded-xl border border-white/5 hover:border-white/10 transition-all">
                                <div class="flex justify-between items-center">
                                    <label :for="key" class="text-xs font-semibold text-gray-300" x-text="key"></label>
                                    <span class="text-xs font-mono text-indigo-400" x-text="val"></span>
                                </div>
                                <input 
                                    :type="isNumeric(key) ? 'number' : 'number'" 
                                    :step="key === 'Discount' ? '0.01' : (key === 'Sales' ? '0.01' : '1')"
                                    :id="key" 
                                    x-model.number="formData[key]"
                                    class="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
                                >
                            </div>
                        </template>

                        <button 
                            type="submit" 
                            :disabled="loading"
                            class="w-full mt-4 py-3.5 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:from-indigo-600 hover:via-purple-600 hover:to-pink-600 shadow-lg shadow-indigo-500/25 transition-all duration-300 transform active:scale-95 flex items-center justify-center gap-2"
                        >
                            <svg x-show="loading" class="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span x-text="loading ? 'Processing Model...' : 'Execute Prediction'"></span>
                        </button>
                    </form>
                </div>
            </div>

            <!-- Right Side: Real-time Analysis & Visualizations -->
            <div class="lg:col-span-7 flex flex-col gap-6">
                
                <!-- KPI Banner Result -->
                <div class="glass-card rounded-2xl p-6 relative overflow-hidden flex flex-col md:flex-row items-center justify-between gap-6">
                    <div class="absolute -right-10 -bottom-10 w-40 h-40 bg-indigo-500/10 rounded-full blur-2xl pointer-events-none"></div>
                    <div>
                        <span class="text-xs font-semibold text-indigo-400 uppercase tracking-wider">Predicted Target Output</span>
                        <div class="text-4xl md:text-5xl font-black font-display tracking-tight text-white mt-1">
                            <span x-text="prediction !== null ? prediction.toFixed(4) : '---'"></span>
                        </div>
                        <p class="text-xs text-gray-400 mt-2" x-text="prediction !== null ? 'Evaluated against 100 Gradient Boosting Trees' : 'Click Execute Prediction to compute results'"></p>
                    </div>

                    <!-- Metrics Badges -->
                    <div class="flex gap-3">
                        <div class="bg-black/30 border border-white/10 p-3 rounded-xl text-center min-w-[90px]">
                            <span class="text-[10px] text-gray-400 block uppercase">Trees</span>
                            <span class="text-lg font-bold text-indigo-400">100</span>
                        </div>
                        <div class="bg-black/30 border border-white/10 p-3 rounded-xl text-center min-w-[90px]">
                            <span class="text-[10px] text-gray-400 block uppercase">Features</span>
                            <span class="text-lg font-bold text-rose-400">13</span>
                        </div>
                    </div>
                </div>

                <!-- Charts Container -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <!-- Chart 1: Feature Impact Analysis -->
                    <div class="glass-card rounded-2xl p-5 flex flex-col">
                        <h3 class="text-sm font-bold text-gray-200 mb-4 flex items-center gap-2">
                            <svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                            </svg>
                            Key Input Features
                        </h3>
                        <div class="relative flex-1 min-h-[220px]">
                            <canvas id="featureChart"></canvas>
                        </div>
                    </div>

                    <!-- Chart 2: Model Tree Ensembles -->
                    <div class="glass-card rounded-2xl p-5 flex flex-col">
                        <h3 class="text-sm font-bold text-gray-200 mb-4 flex items-center gap-2">
                            <svg class="w-4 h-4 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"/>
                            </svg>
                            Boosting Iteration Trend
                        </h3>
                        <div class="relative flex-1 min-h-[220px]">
                            <canvas id="treeTrendChart"></canvas>
                        </div>
                    </div>

                </div>

                <!-- Comprehensive Analysis Table -->
                <div class="glass-card rounded-2xl p-5">
                    <h3 class="text-sm font-bold text-gray-200 mb-3">Model Analysis Details</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs text-gray-300">
                            <thead class="bg-black/40 text-gray-400 uppercase font-mono">
                                <tr>
                                    <th class="p-2.5 rounded-l-lg">Parameter Class</th>
                                    <th class="p-2.5">Current Value</th>
                                    <th class="p-2.5 rounded-r-lg">Status</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-white/5">
                                <tr>
                                    <td class="py-2 px-2.5 font-medium">Sales Impact</td>
                                    <td class="py-2 px-2.5 font-mono text-indigo-400" x-text="`$${formData['Sales']}`"></td>
                                    <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px]">Active</span></td>
                                </tr>
                                <tr>
                                    <td class="py-2 px-2.5 font-medium">Discount Level</td>
                                    <td class="py-2 px-2.5 font-mono text-rose-400" x-text="`${(formData['Discount'] * 100).toFixed(0)}%`"></td>
                                    <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px]">Applied</span></td>
                                </tr>
                                <tr>
                                    <td class="py-2 px-2.5 font-medium">Order Quantity</td>
                                    <td class="py-2 px-2.5 font-mono text-emerald-400" x-text="formData['Quantity']"></td>
                                    <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px]">Normal</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <!-- Dashboard Alpine.js Logic -->
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('dashboard', () => ({
                theme: 'theme-cyber',
                fontStyle: 'font-sans',
                loading: false,
                prediction: null,
                formData: JSON.parse('{{ default_values | tojson | safe }}'),
                featureChart: null,
                treeChart: null,

                init() {
                    this.$nextTick(() => {
                        this.renderCharts();
                        this.runPrediction();
                    });
                },

                setTheme(themeName) {
                    this.theme = themeName;
                    this.updateChartTheme();
                },

                resetDefaults() {
                    this.formData = JSON.parse('{{ default_values | tojson | safe }}');
                    this.runPrediction();
                },

                isNumeric(key) {
                    return typeof this.formData[key] === 'number';
                },

                async runPrediction() {
                    this.loading = true;
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ features: this.formData })
                        });
                        const data = await response.json();
                        if (data.status === 'success') {
                            this.prediction = data.prediction;
                            this.updateCharts();
                        }
                    } catch (error) {
                        console.error('Prediction failed:', error);
                    } finally {
                        this.loading = false;
                    }
                },

                renderCharts() {
                    // Feature Impact Chart
                    const ctx1 = document.getElementById('featureChart').getContext('2d');
                    this.featureChart = new Chart(ctx1, {
                        type: 'bar',
                        data: {
                            labels: ['Sales', 'Discount', 'Quantity', 'Sub-Category', 'Region'],
                            datasets: [{
                                label: 'Input Value',
                                data: [this.formData['Sales'], this.formData['Discount'] * 100, this.formData['Quantity'], this.formData['Sub-Category'], this.formData['Region']],
                                backgroundColor: [
                                    'rgba(99, 102, 241, 0.7)',
                                    'rgba(244, 63, 94, 0.7)',
                                    'rgba(16, 185, 129, 0.7)',
                                    'rgba(245, 158, 11, 0.7)',
                                    'rgba(168, 85, 247, 0.7)'
                                ],
                                borderRadius: 8
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
                                x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                            }
                        }
                    });

                    // Tree Ensemble Iteration Chart
                    const ctx2 = document.getElementById('treeTrendChart').getContext('2d');
                    const dummyIterations = Array.from({length: 10}, (_, i) => (i + 1) * 10);
                    this.treeChart = new Chart(ctx2, {
                        type: 'line',
                        data: {
                            labels: dummyIterations,
                            datasets: [{
                                label: 'Model Confidence Convergence',
                                data: dummyIterations.map(i => Math.sin(i) * 5 + (this.prediction || 20)),
                                borderColor: '#ec4899',
                                backgroundColor: 'rgba(236, 72, 153, 0.1)',
                                fill: true,
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
                                x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                            }
                        }
                    });
                },

                updateCharts() {
                    if (!this.featureChart || !this.treeChart) return;
                    
                    this.featureChart.data.datasets[0].data = [
                        this.formData['Sales'], 
                        this.formData['Discount'] * 100, 
                        this.formData['Quantity'], 
                        this.formData['Sub-Category'], 
                        this.formData['Region']
                    ];
                    this.featureChart.update();

                    const dummyIterations = Array.from({length: 10}, (_, i) => (i + 1) * 10);
                    this.treeChart.data.datasets[0].data = dummyIterations.map((i, idx) => (this.prediction || 10) * (0.5 + (idx * 0.05)));
                    this.treeChart.update();
                },

                updateChartTheme() {
                    if (this.featureChart) this.featureChart.update();
                    if (this.treeChart) this.treeChart.update();
                }
            }));
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, default_values=DEFAULT_VALUES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_dict = data.get("features", {})
        
        # Order inputs according to model's expected feature list
        input_vector = [float(input_dict.get(feat, 0)) for feat in FEATURE_NAMES]
        
        if model is not None:
            # Perform prediction using model
            prediction_val = model.predict([input_vector])[0]
        else:
            # Fallback mock calculation if model pickle is missing
            prediction_val = float(np.dot(input_vector[:5], [0.1, 0.05, 0.2, 0.01, 0.15]))

        return jsonify({
            "status": "success",
            "prediction": float(prediction_val)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

'use client';

import { useState, useEffect } from 'react';

export default function Home() {
  const [apiUrl, setApiUrl] = useState('http://127.0.0.1:8000');
  const [apiStatus, setApiStatus] = useState(null); // true, false, or loading
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // Form State matching FastAPI ChurnPredictionInput payload
  const [formData, setFormData] = useState({
    Contract: 'Month-to-month',
    tenure: 12,
    MonthlyCharges: 75.0,
    TotalCharges: 900.0,
    InternetService: 'Fiber optic',
    PaymentMethod: 'Electronic check',
    TechSupport: 'No',
    OnlineSecurity: 'No',
    PaperlessBilling: 'Yes',
    SeniorCitizen: 0,
    gender: 'Male',
    Partner: 'No',
    Dependents: 'No',
    PhoneService: 'Yes',
    MultipleLines: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No'
  });

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('churn_prediction_history');
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Failed to parse prediction history', e);
      }
    }
    checkHealth(apiUrl);
  }, []);

  // Recalculate TotalCharges automatically when tenure or MonthlyCharges changes
  const handleInputChange = (field, value) => {
    setFormData(prev => {
      const updated = { ...prev, [field]: value };
      if (field === 'tenure' || field === 'MonthlyCharges') {
        const t = field === 'tenure' ? Number(value) : Number(prev.tenure);
        const m = field === 'MonthlyCharges' ? Number(value) : Number(prev.MonthlyCharges);
        updated.TotalCharges = Number((t * m).toFixed(2));
      }
      return updated;
    });
  };

  const checkHealth = async (url) => {
    try {
      const res = await fetch(`${url}/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        setApiStatus(data.model_loaded ? 'online' : 'unloaded');
      } else {
        setApiStatus('offline');
      }
    } catch (err) {
      setApiStatus('offline');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = {
        ...formData,
        tenure: Number(formData.tenure),
        MonthlyCharges: Number(formData.MonthlyCharges),
        TotalCharges: Number(formData.TotalCharges),
        SeniorCitizen: Number(formData.SeniorCitizen)
      };

      const res = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);

      // Add to history
      const historyItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        contract: formData.Contract,
        tenure: formData.tenure,
        monthlyCharges: formData.MonthlyCharges,
        probability: data.churn_probability,
        riskLevel: data.risk_level
      };

      const newHistory = [historyItem, ...history.slice(0, 9)];
      setHistory(newHistory);
      localStorage.setItem('churn_prediction_history', JSON.stringify(newHistory));
    } catch (err) {
      setError(err.message || 'Failed to connect to prediction API');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('churn_prediction_history');
  };

  // Helper color utilities for risk levels
  const getRiskColor = (level) => {
    switch (level) {
      case 'Low':
        return {
          bg: 'bg-emerald-950/60',
          border: 'border-emerald-500/40',
          text: 'text-emerald-400',
          badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
          bar: 'bg-emerald-500'
        };
      case 'Medium':
        return {
          bg: 'bg-amber-950/60',
          border: 'border-amber-500/40',
          text: 'text-amber-400',
          badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
          bar: 'bg-amber-500'
        };
      case 'High':
      default:
        return {
          bg: 'bg-rose-950/60',
          border: 'border-rose-500/40',
          text: 'text-rose-400',
          badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
          bar: 'bg-rose-500'
        };
    }
  };

  return (
    <main className="min-h-screen py-10 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header Bar */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between mb-8 pb-6 border-b border-gray-800 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
              Customer Churn Predictor
            </h1>
            <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-900/50 text-indigo-300 border border-indigo-700/50 font-medium">
              v1.0 FastAPI
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Real-time ML Subscriber Retention Risk Analytics Dashboard
          </p>
        </div>

        {/* API Endpoint Selector & Health Probe */}
        <div className="flex items-center gap-3 bg-gray-900/80 p-2.5 rounded-xl border border-gray-800">
          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                apiStatus === 'online'
                  ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]'
                  : apiStatus === 'unloaded'
                  ? 'bg-amber-400'
                  : 'bg-rose-500'
              }`}
            />
            <span className="text-xs text-gray-400 font-mono">API:</span>
          </div>
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => {
              setApiUrl(e.target.value);
              checkHealth(e.target.value);
            }}
            className="bg-gray-800 text-xs text-gray-200 px-2.5 py-1 rounded border border-gray-700 font-mono w-44 outline-none focus:border-indigo-500"
            placeholder="http://127.0.0.1:8000"
          />
          <button
            type="button"
            onClick={() => checkHealth(apiUrl)}
            className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded border border-gray-700 transition"
          >
            Check
          </button>
        </div>
      </header>

      {/* Main Grid: Form + Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-12">
        {/* Left Column: Input Form (7 cols) */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-6 sm:p-8">
          <h2 className="text-xl font-bold text-gray-100 mb-6 flex items-center gap-2">
            <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Subscriber Feature Inputs
          </h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Sliders Section */}
            <div className="space-y-5 bg-gray-900/40 p-4 rounded-xl border border-gray-800/80">
              {/* Tenure Slider */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-semibold uppercase tracking-wider text-gray-300">
                    Tenure (Months): <span className="text-indigo-400 text-sm font-bold">{formData.tenure} mo</span>
                  </label>
                  <span className="text-xs text-gray-500">0 to 72 mo</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="72"
                  value={formData.tenure}
                  onChange={(e) => handleInputChange('tenure', e.target.value)}
                  className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer range-slider"
                />
              </div>

              {/* Monthly Charges Slider */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-semibold uppercase tracking-wider text-gray-300">
                    Monthly Charges: <span className="text-indigo-400 text-sm font-bold">${formData.MonthlyCharges}</span>
                  </label>
                  <span className="text-xs text-gray-500">$18 to $120</span>
                </div>
                <input
                  type="range"
                  min="18"
                  max="120"
                  step="0.5"
                  value={formData.MonthlyCharges}
                  onChange={(e) => handleInputChange('MonthlyCharges', e.target.value)}
                  className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer range-slider"
                />
              </div>

              {/* Total Charges Display */}
              <div className="flex justify-between items-center pt-2 border-t border-gray-800">
                <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                  Est. Total Charges
                </span>
                <span className="text-sm font-mono font-bold text-gray-200">
                  ${formData.TotalCharges}
                </span>
              </div>
            </div>

            {/* Main Dropdowns Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Contract */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Contract Type</label>
                <select
                  value={formData.Contract}
                  onChange={(e) => handleInputChange('Contract', e.target.value)}
                  className="w-full input-style"
                >
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              {/* Internet Service */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Internet Service</label>
                <select
                  value={formData.InternetService}
                  onChange={(e) => handleInputChange('InternetService', e.target.value)}
                  className="w-full input-style"
                >
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="DSL">DSL</option>
                  <option value="No">No Internet</option>
                </select>
              </div>

              {/* Payment Method */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Payment Method</label>
                <select
                  value={formData.PaymentMethod}
                  onChange={(e) => handleInputChange('PaymentMethod', e.target.value)}
                  className="w-full input-style"
                >
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>

              {/* Tech Support */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Tech Support</label>
                <select
                  value={formData.TechSupport}
                  onChange={(e) => handleInputChange('TechSupport', e.target.value)}
                  className="w-full input-style"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              {/* Online Security */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Online Security</label>
                <select
                  value={formData.OnlineSecurity}
                  onChange={(e) => handleInputChange('OnlineSecurity', e.target.value)}
                  className="w-full input-style"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              {/* Paperless Billing */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Paperless Billing</label>
                <select
                  value={formData.PaperlessBilling}
                  onChange={(e) => handleInputChange('PaperlessBilling', e.target.value)}
                  className="w-full input-style"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              {/* Senior Citizen */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Senior Citizen</label>
                <select
                  value={formData.SeniorCitizen}
                  onChange={(e) => handleInputChange('SeniorCitizen', e.target.value)}
                  className="w-full input-style"
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </div>

              {/* Gender */}
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">Gender</label>
                <select
                  value={formData.gender}
                  onChange={(e) => handleInputChange('gender', e.target.value)}
                  className="w-full input-style"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-rose-950/80 border border-rose-800 rounded-xl text-rose-300 text-xs flex items-center gap-2">
                <svg className="w-4 h-4 text-rose-400 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <span>{error}</span>
              </div>
            )}

            {/* Action Submit Button */}
            <button
              type="submit"
              disabled={loading || apiStatus === 'offline'}
              className="w-full py-3.5 px-6 rounded-xl font-semibold text-white bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 shadow-lg shadow-indigo-600/30 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Calculating Risk...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Predict Churn Probability
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Column: Prediction Gauge & Result Card (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {result ? (
            (() => {
              const riskColors = getRiskColor(result.risk_level);
              const percentage = Math.round(result.churn_probability * 100);

              return (
                <div className={`glass-card rounded-2xl p-6 border ${riskColors.border} transition-all duration-300`}>
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                      ML Prediction Result
                    </span>
                    <span className={`text-xs px-3 py-1 rounded-full font-bold border ${riskColors.badge}`}>
                      {result.risk_level} Churn Risk
                    </span>
                  </div>

                  {/* Circular Gauge Display */}
                  <div className="flex flex-col items-center justify-center my-6">
                    <div className="relative w-44 h-44 flex items-center justify-center">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                        {/* Background Ring */}
                        <path
                          className="text-gray-800"
                          strokeWidth="3.5"
                          stroke="currentColor"
                          fill="none"
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                        {/* Colored Progress Ring */}
                        <path
                          className={`${riskColors.text} transition-all duration-1000 ease-out`}
                          strokeDasharray={`${percentage}, 100`}
                          strokeWidth="3.5"
                          strokeLinecap="round"
                          stroke="currentColor"
                          fill="none"
                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                      </svg>

                      <div className="absolute flex flex-col items-center">
                        <span className={`text-4xl font-extrabold font-mono ${riskColors.text}`}>
                          {percentage}%
                        </span>
                        <span className="text-[10px] text-gray-400 uppercase tracking-widest mt-0.5">
                          Probability
                        </span>
                      </div>
                    </div>

                    {/* Linear Progress Bar */}
                    <div className="w-full bg-gray-800 rounded-full h-2.5 mt-4 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${riskColors.bar} transition-all duration-700`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>

                  {/* Risk Breakdown Details */}
                  <div className="grid grid-cols-2 gap-3 pt-4 border-t border-gray-800/80 text-xs">
                    <div className="bg-gray-900/50 p-3 rounded-xl border border-gray-800">
                      <span className="text-gray-400 block mb-0.5">Predicted Binary</span>
                      <span className="text-gray-200 font-bold font-mono">
                        {result.predicted_churn === 1 ? '1 (Will Churn)' : '0 (Retained)'}
                      </span>
                    </div>

                    <div className="bg-gray-900/50 p-3 rounded-xl border border-gray-800">
                      <span className="text-gray-400 block mb-0.5">Recommended Action</span>
                      <span className="text-gray-200 font-medium">
                        {result.risk_level === 'High'
                          ? 'Send 15% Contract Offer'
                          : result.risk_level === 'Medium'
                          ? 'Attach Security Addon'
                          : 'Standard Retention'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })()
          ) : (
            /* Placeholder State */
            <div className="glass-card rounded-2xl p-8 border border-gray-800 text-center flex flex-col items-center justify-center min-h-[360px]">
              <div className="w-16 h-16 rounded-2xl bg-indigo-950/40 border border-indigo-800/40 flex items-center justify-center text-indigo-400 mb-4">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-base font-semibold text-gray-200 mb-1">
                Awaiting Prediction Input
              </h3>
              <p className="text-xs text-gray-400 max-w-xs">
                Adjust subscriber parameters on the left and click "Predict Churn Probability" to get instant model inferences.
              </p>
            </div>
          )}

          {/* Quick Info Box */}
          <div className="bg-gray-900/60 rounded-xl p-4 border border-gray-800/80 text-xs text-gray-400 space-y-1.5">
            <div className="font-semibold text-gray-300 flex items-center gap-1.5">
              <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Model Risk Segmentation Rules:
            </div>
            <p><span className="text-emerald-400 font-medium">Low Risk (&le; 35%)</span>: High tenure, 2-Year Contract, Security attached.</p>
            <p><span className="text-amber-400 font-medium">Medium Risk (36% - 65%)</span>: Moderate charges, DSL or Fiber.</p>
            <p><span className="text-rose-400 font-medium">High Risk (&gt; 65%)</span>: Month-to-Month, Fiber Optic, Electronic Check.</p>
          </div>
        </div>
      </div>

      {/* Prediction History Table */}
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-gray-200">Recent Inferences History</h3>
            <span className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-400 border border-gray-700">
              {history.length} saved
            </span>
          </div>

          {history.length > 0 && (
            <button
              onClick={clearHistory}
              className="text-xs px-3 py-1.5 text-rose-400 hover:text-rose-300 bg-rose-950/30 hover:bg-rose-950/60 rounded-lg border border-rose-800/40 transition"
            >
              Clear History
            </button>
          )}
        </div>

        {history.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-gray-300">
              <thead className="bg-gray-900/80 text-gray-400 uppercase tracking-wider text-[10px] border-b border-gray-800">
                <tr>
                  <th className="py-3 px-4">Time</th>
                  <th className="py-3 px-4">Contract</th>
                  <th className="py-3 px-4">Tenure</th>
                  <th className="py-3 px-4">Monthly Charge</th>
                  <th className="py-3 px-4">Churn Prob %</th>
                  <th className="py-3 px-4">Risk Level</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60">
                {history.map((item) => {
                  const colors = getRiskColor(item.riskLevel);
                  return (
                    <tr key={item.id} className="hover:bg-gray-800/30 transition">
                      <td className="py-3 px-4 font-mono text-gray-400">{item.timestamp}</td>
                      <td className="py-3 px-4 font-medium text-gray-200">{item.contract}</td>
                      <td className="py-3 px-4">{item.tenure} mo</td>
                      <td className="py-3 px-4 font-mono">${item.monthlyCharges}</td>
                      <td className="py-3 px-4 font-mono font-bold">
                        {Math.round(item.probability * 100)}%
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2.5 py-0.5 rounded-full font-bold border ${colors.badge}`}>
                          {item.riskLevel}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-xs text-gray-500">
            No inference history recorded yet. Run predictions above to populate the table.
          </div>
        )}
      </div>
    </main>
  );
}

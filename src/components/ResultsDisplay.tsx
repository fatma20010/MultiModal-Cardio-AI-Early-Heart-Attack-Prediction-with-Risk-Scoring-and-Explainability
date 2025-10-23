import { AlertCircle, CheckCircle, TrendingUp, Activity, Brain, Download } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { DetailedPredictionResponse, downloadReport } from '@/services/api';
import { cn } from '@/lib/utils';
import { toast } from 'react-hot-toast';

interface ResultsDisplayProps {
  results: DetailedPredictionResponse | null;
  onReset: () => void;
}

export const ResultsDisplay = ({ results, onReset }: ResultsDisplayProps) => {
  const [isDownloading, setIsDownloading] = useState(false);

  if (!results) return null;

  const handleDownloadReport = async () => {
    setIsDownloading(true);
    const loadingToast = toast.loading('Generating PDF report...');
    
    try {
      const blob = await downloadReport(results);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `Cardiac_Assessment_Report_${Date.now()}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      toast.success('Report downloaded successfully!', { id: loadingToast });
    } catch (error) {
      toast.error('Failed to download report. Please try again.', { id: loadingToast });
    } finally {
      setIsDownloading(false);
    }
  };

  const getRiskColor = (level: number) => {
    if (level === 1 || level === 2) return 'text-success';
    if (level === 3) return 'text-warning';
    if (level === 4 || level === 5) return 'text-danger';
    return 'text-muted-foreground';
  };

  const getRiskBg = (level: number) => {
    if (level === 1 || level === 2) return 'bg-success/10 border-success';
    if (level === 3) return 'bg-warning/10 border-warning';
    if (level === 4 || level === 5) return 'bg-danger/10 border-danger';
    return 'bg-muted';
  };

  const modalityData = results.explainability ? [
    { name: 'ECG', value: results.explainability.modality_contributions.ecg.percentage / 100, color: 'hsl(var(--medical-blue))' },
    { name: 'PCG', value: results.explainability.modality_contributions.pcg.percentage / 100, color: 'hsl(var(--medical-green))' },
    { name: 'Clinical', value: results.explainability.modality_contributions.clinical.percentage / 100, color: 'hsl(var(--medical-pink))' }
  ] : [];

  const featureData = results.explainability?.top_clinical_features.slice(0, 5).map(f => ({
    name: f.feature.toUpperCase(),
    importance: Number(f.importance) * 100,
    interpretation: f.interpretation
  })) || [];

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Main Risk Card */}
      <div className={cn(
        "gradient-card rounded-2xl p-8 shadow-medical border-2",
        getRiskBg(results.risk_assessment.level)
      )}>
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className={cn("p-4 rounded-xl", getRiskBg(results.risk_assessment.level))}>
              {results.risk_assessment.level <= 2 ? (
                <CheckCircle className="w-8 h-8 text-success" />
              ) : (
                <AlertCircle className="w-8 h-8 text-danger" />
              )}
            </div>
            <div>
              <h3 className="text-2xl font-bold text-foreground">Risk Assessment Complete</h3>
              <p className="text-muted-foreground">Based on multimodal analysis</p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button 
              onClick={handleDownloadReport} 
              variant="default" 
              className="rounded-xl"
              disabled={isDownloading}
            >
              <Download className="w-4 h-4 mr-2" />
              {isDownloading ? 'Generating...' : 'Download Report'}
            </Button>
            <Button onClick={onReset} variant="outline" className="rounded-xl">
              New Assessment
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <p className="text-sm text-muted-foreground mb-2">Cardiac Risk Probability</p>
              <div className={cn("text-5xl font-bold", getRiskColor(results.risk_assessment.level))}>
                {(results.probability * 100).toFixed(1)}%
              </div>
            </div>
            
            <div>
              <p className="text-sm text-muted-foreground mb-2">Risk Level</p>
              <div className={cn(
                "inline-flex items-center gap-2 px-4 py-2 rounded-xl font-semibold",
                getRiskBg(results.risk_assessment.level)
              )}>
                <TrendingUp className="w-5 h-5" />
                {results.risk_assessment.category}
              </div>
            </div>

            <div>
              <p className="text-sm text-muted-foreground mb-2">Level</p>
              <p className="font-medium text-foreground">Level {results.risk_assessment.level} of 5</p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <p className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Interpretation
              </p>
              <p className="text-foreground leading-relaxed">{results.risk_assessment.interpretation}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="gradient-card rounded-2xl p-6 shadow-medical border border-border">
          <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-primary" />
            Recommendation
          </h4>
          <p className="text-foreground leading-relaxed">{results.risk_assessment.recommendation}</p>
        </div>

        <div className="gradient-card rounded-2xl p-6 shadow-medical border border-border">
          <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-accent" />
            Action Required
          </h4>
          <p className="text-foreground leading-relaxed">{results.risk_assessment.action}</p>
        </div>
      </div>

      {/* Explainability Section */}
      {results.explainability && (
        <div className="gradient-card rounded-2xl p-8 shadow-medical border border-border">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-primary/10 p-3 rounded-xl">
              <Brain className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-foreground">AI Explainability</h3>
              <p className="text-sm text-muted-foreground">Understanding the prediction</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Modality Contributions */}
            <div>
              <h4 className="font-semibold text-foreground mb-4">Modality Contributions</h4>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={modalityData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }: any) => `${name}: ${(Number(value) * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {modalityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: any) => `${(Number(value) * 100).toFixed(1)}%`}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Top Clinical Features */}
            <div>
              <h4 className="font-semibold text-foreground mb-4">Top Clinical Features</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={featureData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" />
                  <YAxis dataKey="name" type="category" width={100} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip 
                    formatter={(value: any) => `${Number(value).toFixed(1)}%`}
                    contentStyle={{ 
                      background: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confidence Assessment */}
          <div className="mt-8">
            <div className="bg-muted/20 rounded-xl p-6 border border-border">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="w-5 h-5 text-primary" />
                <h4 className="font-semibold text-foreground">Confidence Assessment</h4>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Confidence Level:</span>
                  <span className={cn(
                    "px-3 py-1 rounded-full text-sm font-medium",
                    results.explainability.confidence_assessment.level === 'high' 
                      ? 'bg-success/20 text-success' 
                      : results.explainability.confidence_assessment.level === 'moderate'
                      ? 'bg-warning/20 text-warning'
                      : 'bg-muted text-foreground'
                  )}>
                    {results.explainability.confidence_assessment.level.toUpperCase()}
                  </span>
                </div>
                <p className="text-sm text-foreground leading-relaxed">
                  {results.explainability.confidence_assessment.message}
                </p>
                <div className="pt-2 border-t border-border">
                  <p className="text-sm text-muted-foreground italic">
                    💡 {results.explainability.summary}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

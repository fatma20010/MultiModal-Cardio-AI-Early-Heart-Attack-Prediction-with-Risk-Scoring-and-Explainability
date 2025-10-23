import { useState, useRef } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { Header } from '@/components/Header';
import { Hero } from '@/components/Hero';
import { UploadCard } from '@/components/UploadCard';
import { ClinicalForm } from '@/components/ClinicalForm';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { Footer } from '@/components/Footer';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { predictRisk, ClinicalData, DetailedPredictionResponse } from '@/services/api';

const Index = () => {
  const [ecgFile, setEcgFile] = useState<File | null>(null);
  const [pcgFile, setPcgFile] = useState<File | null>(null);
  const [detailedAnalysis, setDetailedAnalysis] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<DetailedPredictionResponse | null>(null);
  
  const assessmentRef = useRef<HTMLDivElement>(null);

  const scrollToAssessment = () => {
    assessmentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const handleSubmit = async (clinicalData: ClinicalData) => {
    if (!ecgFile || !pcgFile) {
      toast.error('Please upload both ECG image and PCG audio files');
      return;
    }

    setIsLoading(true);
    const loadingToast = toast.loading('Analyzing cardiac data...');

    try {
      const response = await predictRisk(ecgFile, pcgFile, clinicalData, detailedAnalysis);
      setResults(response as DetailedPredictionResponse);
      toast.success('Analysis complete!', { id: loadingToast });
      
      // Scroll to results
      setTimeout(() => {
        const resultsElement = document.getElementById('results');
        resultsElement?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Analysis failed. Please try again.', {
        id: loadingToast
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setEcgFile(null);
    setPcgFile(null);
    setResults(null);
    scrollToAssessment();
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'hsl(var(--card))',
            color: 'hsl(var(--foreground))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '12px',
            padding: '16px',
          },
          success: {
            iconTheme: {
              primary: 'hsl(var(--success))',
              secondary: 'white',
            },
          },
          error: {
            iconTheme: {
              primary: 'hsl(var(--danger))',
              secondary: 'white',
            },
          },
        }}
      />
      
      <Header />
      
      <main className="flex-1 pt-24">
        <Hero onGetStarted={scrollToAssessment} />
        
        <div className="container mx-auto px-4 py-12" ref={assessmentRef}>
          <div className="max-w-7xl mx-auto space-y-8">
            {/* Analysis Type Toggle */}
            <div className="flex items-center justify-center gap-3 p-4 bg-card rounded-xl border border-border shadow-sm">
              <Label htmlFor="analysis-type" className="text-foreground font-medium">
                Basic Analysis
              </Label>
              <Switch
                id="analysis-type"
                checked={detailedAnalysis}
                onCheckedChange={setDetailedAnalysis}
              />
              <Label htmlFor="analysis-type" className="text-foreground font-medium">
                Detailed Analysis (with AI Explainability)
              </Label>
            </div>

            {/* File Uploads */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <UploadCard
                type="ecg"
                title="ECG Image"
                description="Upload electrocardiogram image"
                accept=".png,.jpg,.jpeg"
                maxSize={5}
                file={ecgFile}
                onFileChange={setEcgFile}
              />
              
              <UploadCard
                type="pcg"
                title="PCG Audio"
                description="Upload phonocardiogram recording"
                accept=".wav,.mp3"
                maxSize={10}
                file={pcgFile}
                onFileChange={setPcgFile}
              />
            </div>

            {/* Clinical Form */}
            <ClinicalForm onSubmit={handleSubmit} isLoading={isLoading} />

            {/* Results */}
            {results && (
              <div id="results" className="scroll-mt-24">
                <ResultsDisplay results={results} onReset={handleReset} />
              </div>
            )}
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;

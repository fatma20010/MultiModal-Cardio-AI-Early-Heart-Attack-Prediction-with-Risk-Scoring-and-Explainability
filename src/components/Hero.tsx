import { Heart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import heroImage from '@/assets/hero-medical.jpg';

interface HeroProps {
  onGetStarted: () => void;
}

export const Hero = ({ onGetStarted }: HeroProps) => {
  return (
    <section className="relative gradient-hero py-20 px-4 overflow-hidden">
      <div className="absolute inset-0 opacity-20">
        <img 
          src={heroImage} 
          alt="Medical background" 
          className="w-full h-full object-cover"
        />
      </div>
      
      <div className="container mx-auto relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center justify-center mb-6">
            <div className="bg-primary/20 p-6 rounded-full shadow-glow">
              <Heart className="w-16 h-16 text-primary animate-float" fill="currentColor" />
            </div>
          </div>
          
          <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6 leading-tight">
            AI-Powered Cardiac Health Assessment
          </h2>
          
          <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
            Upload your ECG image, PCG audio recording, and clinical data for comprehensive, 
            AI-driven cardiac risk analysis with actionable insights
          </p>
          
          <Button 
            size="lg"
            onClick={onGetStarted}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-8 py-6 text-lg rounded-xl shadow-medical transition-smooth hover:shadow-glow"
          >
            Start Assessment
            <Heart className="ml-2 w-5 h-5" />
          </Button>
          
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-card/80 backdrop-blur-sm p-6 rounded-xl shadow-medical">
              <div className="text-3xl font-bold text-primary mb-2">3</div>
              <p className="text-sm text-muted-foreground">Data Modalities</p>
            </div>
            <div className="bg-card/80 backdrop-blur-sm p-6 rounded-xl shadow-medical">
              <div className="text-3xl font-bold text-secondary mb-2">AI</div>
              <p className="text-sm text-muted-foreground">Advanced Analysis</p>
            </div>
            <div className="bg-card/80 backdrop-blur-sm p-6 rounded-xl shadow-medical">
              <div className="text-3xl font-bold text-accent mb-2">100%</div>
              <p className="text-sm text-muted-foreground">Confidential</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

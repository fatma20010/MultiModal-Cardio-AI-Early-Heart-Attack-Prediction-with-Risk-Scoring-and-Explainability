import { HeartPulse } from 'lucide-react';

export const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-card/95 backdrop-blur-sm border-b border-border shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-primary/10 p-2 rounded-xl">
              <HeartPulse className="w-8 h-8 text-primary animate-heartbeat" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-primary">CardioAI Diagnosis</h1>
              <p className="text-sm text-muted-foreground">Multimodal Cardiac Risk Assessment</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-4">
            <div className="text-right">
              <p className="text-xs text-muted-foreground">Powered by</p>
              <p className="text-sm font-semibold text-foreground">Advanced AI Technology</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

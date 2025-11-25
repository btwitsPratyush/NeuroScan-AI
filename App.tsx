import { useRef } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import About from './components/About';
import Detection from './components/Detection';
import ImageGallery from './components/ImageGallery';
import Footer from './components/Footer';

function App() {
    const startDetectionHandlerRef = useRef<(() => void) | null>(null);

    const handleStartDetectionReady = (handler: () => void) => {
        startDetectionHandlerRef.current = handler;
    };

    const handleGetStartedClick = () => {
        if (startDetectionHandlerRef.current) {
            startDetectionHandlerRef.current();
        }
    };

    return (
        <div className="min-h-screen bg-white dark:bg-gray-900">
            <Header onGetStartedClick={handleGetStartedClick} />
            <Hero onStartDetectionReady={handleStartDetectionReady} />
            <About />
            <Detection />
            <ImageGallery />
            <Footer />
        </div>
    );
}

export default App;

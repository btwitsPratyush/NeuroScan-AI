import { Brain, Linkedin, Github } from 'lucide-react';

export default function Footer() {
  return (
    <footer id="contact" className="relative bg-gray-950 text-gray-400 overflow-hidden">
      {/* Ambient glow effects */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/5 rounded-full blur-[120px]" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-600/5 rounded-full blur-[120px]" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-12 relative z-10">

        {/* Big headline CTA area */}
        <div className="text-center mb-20">
          <h2 className="text-5xl md:text-7xl font-extrabold text-white tracking-tight leading-tight mb-6">
            Built for the <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Future</span>
          </h2>
          <p className="text-gray-500 text-lg max-w-md mx-auto">
            AI-powered brain tumor detection, designed to assist professionals worldwide.
          </p>
        </div>

        {/* Divider line with gradient */}
        <div className="h-px w-full bg-gradient-to-r from-transparent via-gray-700 to-transparent mb-16" />

        {/* Middle section — brand + nav links + socials */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 items-start mb-16">
          {/* Brand */}
          <div className="flex flex-col items-center md:items-start gap-4">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl blur-sm opacity-40" />
                <div className="relative bg-gradient-to-br from-blue-600 to-cyan-600 p-2.5 rounded-xl">
                  <Brain className="w-6 h-6 text-white" />
                </div>
              </div>
              <span className="text-2xl font-extrabold text-white tracking-tight">
                NeuroScan <span className="font-light text-cyan-400">AI</span>
              </span>
            </div>
            <p className="text-sm text-gray-600 max-w-xs text-center md:text-left">
              Advancing neural diagnostics through deep learning and explainable AI.
            </p>
          </div>

          {/* Navigation */}
          <div className="flex flex-col items-center gap-4">
            <span className="text-xs font-bold uppercase tracking-[0.2em] text-gray-600 mb-2">Navigate</span>
            <div className="flex flex-wrap justify-center gap-x-8 gap-y-3">
              {['Home', 'About', 'Detection', 'Gallery'].map((item) => (
                <a
                  key={item}
                  href={`#${item.toLowerCase()}`}
                  className="text-sm text-gray-500 hover:text-white transition-colors duration-300"
                >
                  {item}
                </a>
              ))}
            </div>
          </div>

          {/* Socials */}
          <div className="flex flex-col items-center md:items-end gap-4">
            <span className="text-xs font-bold uppercase tracking-[0.2em] text-gray-600 mb-2">Connect</span>
            <div className="flex items-center space-x-3">
              <a
                href="https://www.linkedin.com/in/pratyush-linkdin/"
                target="_blank"
                rel="noopener noreferrer"
                className="w-11 h-11 flex items-center justify-center rounded-xl bg-gray-800/50 border border-gray-800 hover:bg-blue-600 hover:border-blue-500 text-gray-500 hover:text-white transition-all duration-300 hover:-translate-y-1"
              >
                <Linkedin className="w-5 h-5" />
              </a>
              <a
                href="https://x.com/btwitsPratyush"
                target="_blank"
                rel="noopener noreferrer"
                className="w-11 h-11 flex items-center justify-center rounded-xl bg-gray-800/50 border border-gray-800 hover:bg-white hover:border-white text-gray-500 hover:text-black transition-all duration-300 hover:-translate-y-1"
              >
                <svg viewBox="0 0 24 24" className="w-5 h-5 fill-current">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </a>
              <a
                href="https://github.com/btwitsPratyush"
                target="_blank"
                rel="noopener noreferrer"
                className="w-11 h-11 flex items-center justify-center rounded-xl bg-gray-800/50 border border-gray-800 hover:bg-gray-700 hover:border-gray-600 text-gray-500 hover:text-white transition-all duration-300 hover:-translate-y-1"
              >
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="h-px w-full bg-gradient-to-r from-transparent via-gray-800 to-transparent mb-8" />
        <div className="flex flex-col md:flex-row justify-between items-center gap-3 text-xs text-gray-600">
          <p>© 2025 NeuroScan AI</p>
          <p>
            Developed with ❤️ by{' '}
            <span className="text-gray-400 font-semibold">PRATYUSH</span>
          </p>
        </div>
      </div>
    </footer>
  );
}

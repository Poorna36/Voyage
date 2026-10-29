import React, { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';

interface NavbarProps {
  activeSection: 'overview' | 'pipeline' | 'sensors' | 'science' | 'team';
  onNavigate: (section: 'overview' | 'pipeline' | 'sensors' | 'science' | 'team') => void;
  onLaunchWorkbench?: () => void;
  onOpenDossier?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeSection,
  onNavigate,
  onLaunchWorkbench,
  onOpenDossier,
}) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = (section: 'overview' | 'pipeline' | 'sensors' | 'science' | 'team') => {
    setMobileMenuOpen(false);
    onNavigate(section);
  };

  return (
    <header
      id="main-navbar"
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-black/95 backdrop-blur-md py-3 border-b border-white/[0.08]'
          : 'bg-black/80 backdrop-blur-sm py-3.5 border-b border-white/[0.05]'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        
        {/* Voyage Logotype */}
        <button
          onClick={() => handleNavClick('overview')}
          className="flex items-center gap-2.5 group shrink-0 text-left focus:outline-none cursor-pointer"
          title="Return to Overview"
          aria-label="Voyage Overview Home"
        >
          <div className="flex items-center gap-2.5">
            <span className="font-logo text-xl sm:text-2xl font-extrabold tracking-[0.08em] text-white group-hover:text-[#2997FF] transition-colors">
              Voyage
            </span>
            <div className="w-4 h-4 rounded-full bg-gradient-to-tr from-[#0F1117] via-slate-300 to-white shadow-[0_0_10px_rgba(255,255,255,0.35)] relative overflow-hidden flex items-center justify-center shrink-0 border border-white/30">
              <div className="absolute inset-0 bg-black/75 rounded-full translate-x-1.5 -translate-y-0.5" />
            </div>
          </div>
        </button>

        {/* Desktop Navigation: OVERVIEW, PIPELINE, SENSORS, SCIENCE, TEAM */}
        <div className="hidden md:flex items-center gap-6 lg:gap-8">
          <nav className="flex items-center space-x-6 lg:space-x-8 text-[12px] font-sans font-medium tracking-wider">
            <button
              onClick={() => handleNavClick('overview')}
              className={`transition-colors duration-200 uppercase relative py-1 cursor-pointer ${
                activeSection === 'overview'
                  ? 'text-white font-semibold'
                  : 'text-[#86868B] hover:text-white'
              }`}
            >
              OVERVIEW
              {activeSection === 'overview' && (
                <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#0071E3] rounded-full" />
              )}
            </button>

            <button
              onClick={() => handleNavClick('pipeline')}
              className={`transition-colors duration-200 uppercase relative py-1 cursor-pointer ${
                activeSection === 'pipeline'
                  ? 'text-white font-semibold'
                  : 'text-[#86868B] hover:text-white'
              }`}
            >
              PIPELINE
              {activeSection === 'pipeline' && (
                <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#0071E3] rounded-full" />
              )}
            </button>

            <button
              onClick={() => handleNavClick('sensors')}
              className={`transition-colors duration-200 uppercase relative py-1 cursor-pointer ${
                activeSection === 'sensors'
                  ? 'text-white font-semibold'
                  : 'text-[#86868B] hover:text-white'
              }`}
            >
              SENSORS
              {activeSection === 'sensors' && (
                <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#0071E3] rounded-full" />
              )}
            </button>

            <button
              onClick={() => handleNavClick('science')}
              className={`transition-colors duration-200 uppercase relative py-1 cursor-pointer ${
                activeSection === 'science'
                  ? 'text-white font-semibold'
                  : 'text-[#86868B] hover:text-white'
              }`}
            >
              SCIENCE
              {activeSection === 'science' && (
                <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#0071E3] rounded-full" />
              )}
            </button>

            <button
              onClick={() => handleNavClick('team')}
              className={`transition-colors duration-200 uppercase relative py-1 cursor-pointer ${
                activeSection === 'team'
                  ? 'text-white font-semibold'
                  : 'text-[#86868B] hover:text-white'
              }`}
            >
              TEAM
              {activeSection === 'team' && (
                <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-[#0071E3] rounded-full" />
              )}
            </button>
          </nav>

          <div className="flex items-center gap-3">
            {onOpenDossier && (
              <button
                onClick={onOpenDossier}
                className="px-4 py-1.5 rounded-full border border-white/20 hover:border-white/50 text-white/80 hover:text-white font-sans font-medium text-[11px] tracking-wider transition-colors cursor-pointer"
              >
                DOSSIER
              </button>
            )}

            {onLaunchWorkbench && (
              <button
                onClick={onLaunchWorkbench}
                className="px-5 py-1.5 rounded-full bg-[#0071E3] hover:bg-[#0077ED] text-white font-sans font-semibold text-[11px] tracking-wider transition-all duration-200 shadow-[0_2px_12px_rgba(0,113,227,0.35)] active:scale-[0.98] cursor-pointer"
              >
                <span>LAUNCH</span>
              </button>
            )}
          </div>
        </div>

        {/* Mobile Hamburger Button */}
        <button
          id="btn-mobile-menu"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 text-white/80 hover:text-white border border-white/10 rounded-full focus:outline-none"
          aria-label="Toggle Navigation Menu"
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>

      </div>

      {/* Mobile Menu Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-black/95 border-b border-white/10 px-6 py-5 space-y-3 font-sans text-xs tracking-widest">
          <button
            onClick={() => handleNavClick('overview')}
            className={`block w-full text-left py-2 border-b border-white/10 uppercase ${
              activeSection === 'overview' ? 'text-[#2997FF] font-bold' : 'text-white'
            }`}
          >
            OVERVIEW
          </button>

          <button
            onClick={() => handleNavClick('pipeline')}
            className={`block w-full text-left py-2 border-b border-white/10 uppercase ${
              activeSection === 'pipeline' ? 'text-[#2997FF] font-bold' : 'text-white'
            }`}
          >
            PIPELINE
          </button>

          <button
            onClick={() => handleNavClick('sensors')}
            className={`block w-full text-left py-2 border-b border-white/10 uppercase ${
              activeSection === 'sensors' ? 'text-[#2997FF] font-bold' : 'text-white'
            }`}
          >
            SENSORS
          </button>

          <button
            onClick={() => handleNavClick('science')}
            className={`block w-full text-left py-2 border-b border-white/10 uppercase ${
              activeSection === 'science' ? 'text-[#2997FF] font-bold' : 'text-white'
            }`}
          >
            SCIENCE
          </button>

          <button
            onClick={() => handleNavClick('team')}
            className={`block w-full text-left py-2 text-[#86868B] hover:text-white uppercase ${
              activeSection === 'team' ? 'text-[#2997FF] font-bold' : ''
            }`}
          >
            TEAM
          </button>

          <div className="pt-2 flex flex-col gap-2">
            {onOpenDossier && (
              <button
                onClick={() => {
                  setMobileMenuOpen(false);
                  onOpenDossier();
                }}
                className="block w-full text-center py-2 rounded-full border border-white/20 text-white/90 font-sans font-medium text-xs tracking-wider"
              >
                SCIENTIFIC DOSSIER
              </button>
            )}

            {onLaunchWorkbench && (
              <button
                onClick={onLaunchWorkbench}
                className="block w-full text-center py-2.5 rounded-full bg-[#0071E3] text-white font-sans font-semibold text-xs tracking-wider shadow-[0_2px_12px_rgba(0,113,227,0.35)]"
              >
                LAUNCH
              </button>
            )}
          </div>
        </div>
      )}
    </header>
  );
};

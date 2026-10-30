import React from 'react';

export const TeamSection: React.FC = () => {
  const teamMembers = [
    { id: '01', name: 'ABHISHEK R P', role: 'RESEARCHER' },
    { id: '02', name: 'ANKITHA V', role: 'RESEARCHER' },
    { id: '03', name: 'MONISH DEV B M', role: 'RESEARCHER' },
    { id: '04', name: 'PIYUSH KUMAR SHUKLA', role: 'RESEARCHER' },
    { id: '05', name: 'POORNACHANDRA B L', role: 'RESEARCHER' },
    { id: '06', name: 'UTKARSH TIWARI', role: 'RESEARCHER' },
  ];

  return (
    <section
      id="team"
      className="relative w-full min-h-[calc(100dvh-65px)] py-20 sm:py-24 md:py-28 lg:py-32 px-4 sm:px-6 md:px-8 lg:px-12 bg-[#07090C] overflow-hidden select-none flex flex-col justify-between"
    >
      {/* Background Subtle Grid & Grain */}
      <div className="absolute inset-0 pointer-events-none grid-lines-lunar opacity-30"></div>
      <div className="absolute inset-0 pointer-events-none lunar-noise-overlay opacity-40"></div>

      <div className="relative z-10 max-w-7xl mx-auto w-full space-y-12 lg:space-y-16 my-auto">
        
        {/* Editorial Section Header */}
        <div className="border-b border-subtle pb-6">
          <p className="font-subheading text-xl sm:text-2xl md:text-3xl text-[#D6C38B] italic mb-2 tracking-wide font-normal">
            "the researchers behind the pixels"
          </p>

          <h2 className="font-headline text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold uppercase tracking-[-0.02em] text-[#E7E3D9]">
            OUR<br />
            <span className="text-[#D6C38B]">TEAM.</span>
          </h2>
        </div>

        {/* Minimal Editorial Layout: Exactly 6 Members in 2 rows of 3 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 lg:gap-x-12 gap-y-10 lg:gap-y-14">
          {teamMembers.map((member) => (
            <div
              key={member.id}
              className="group relative flex flex-col justify-between pt-6 border-t border-subtle hover:border-[#D6C38B]/60 transition-colors duration-300"
            >
              {/* Large Member Number */}
              <div className="flex items-baseline justify-between mb-4">
                <span className="font-headline text-3xl md:text-4xl lg:text-5xl font-light text-[#8B908F]/40 group-hover:text-[#D6C38B] transition-colors duration-300">
                  {member.id}
                </span>
                <span className="w-2 h-2 rounded-full bg-[#8B908F]/30 group-hover:bg-[#D6C38B] transition-colors"></span>
              </div>

              {/* Member Name */}
              <div className="space-y-1.5 mb-4">
                <h3 className="font-headline text-lg sm:text-xl md:text-2xl font-bold text-[#E7E3D9] tracking-tight uppercase group-hover:text-[#D6C38B] transition-colors">
                  {member.name}
                </h3>
                <div className="font-mono-tech text-xs text-[#D6C38B]/90 font-medium tracking-wider uppercase">
                  {member.role}
                </div>
              </div>

              {/* Subtle accent divider */}
              <div className="pt-3 border-t border-subtle/40 font-mono-tech text-[10px] text-[#8B908F] flex items-center justify-between">
                <span>CHANDRAYAAN RESEARCH</span>
                <span className="text-[#8B908F]/60">ECLIPSE</span>
              </div>
            </div>
          ))}
        </div>

      </div>

      {/* Team Page Bottom Bar */}
      <div className="relative z-10 max-w-7xl mx-auto w-full pt-10 flex items-center justify-between border-t border-subtle">
        <div className="font-mono-tech text-[10px] text-[#8B908F] uppercase tracking-widest">
          CHANDRAYAAN • SECTION 03 / 03
        </div>
        <div className="font-mono-tech text-[10px] text-[#D6C38B]/80 tracking-widest uppercase">
          ECLIPSE MISSION RESEARCH
        </div>
      </div>
    </section>
  );
};

import { useState, useEffect } from 'react';

export function useMissionClock() {
  const [timeState, setTimeState] = useState(() => {
    const now = new Date();
    return {
      localStr: now.toLocaleTimeString('en-GB', { hour12: false }),
      dateStr: now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
      tzName: Intl.DateTimeFormat().resolvedOptions().timeZone.includes('Kolkata') || Intl.DateTimeFormat().resolvedOptions().timeZone.includes('Calcutta') ? 'IST' : 'LOCAL',
      utcTimeStr: now.toISOString().substring(11, 19),
    };
  });

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setTimeState({
        localStr: now.toLocaleTimeString('en-GB', { hour12: false }),
        dateStr: now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
        tzName: Intl.DateTimeFormat().resolvedOptions().timeZone.includes('Kolkata') || Intl.DateTimeFormat().resolvedOptions().timeZone.includes('Calcutta') ? 'IST' : 'LOCAL',
        utcTimeStr: now.toISOString().substring(11, 19),
      });
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return timeState;
}

function formatIsoDateLocal(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function getRecentDateRange(days: number): { startDate: string; endDate: string } {
  const safeDays = Math.max(1, Math.floor(days));
  const end = new Date();
  const start = new Date(end);
  start.setDate(end.getDate() - (safeDays - 1));
  return {
    startDate: formatIsoDateLocal(start),
    endDate: formatIsoDateLocal(end),
  };
}

export function dateKeyFromTimestamp(timestamp: number): string {
  return formatIsoDateLocal(new Date(timestamp));
}

export function enumerateDateRange(startDate: string, endDate: string): string[] {
  const result: string[] = [];
  const start = new Date(`${startDate}T00:00:00`);
  const end = new Date(`${endDate}T00:00:00`);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || start > end) {
    return result;
  }
  const cursor = new Date(start);
  while (cursor <= end) {
    result.push(formatIsoDateLocal(cursor));
    cursor.setDate(cursor.getDate() + 1);
  }
  return result;
}
